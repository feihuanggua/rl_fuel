"""PPO 微调 (从 BC 预训练权重初始化)."""
import os
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from fuel_rl.models import Encoder3D
from fuel_rl.models.viewpoint_head import ViewpointActorCritic
from fuel_rl.env.viewpoint_env import ViewpointEnv
from fuel_rl.config import *


class RolloutBuffer:
    def __init__(self):
        self.grids = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, grid, action, logprob, reward, value, done):
        self.grids.append(grid)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()


def load_bc_pretrained(model, bc_path):
    """加载 BC 预训练权重到 Actor-Critic (编码器 + head 共享)."""
    bc_state = torch.load(bc_path, map_location="cpu", weights_only=False)
    model_state = model.state_dict()

    # BC → AC 名称映射
    name_map = {
        "pos_out.weight": "pos_mean.weight",
        "pos_out.bias": "pos_mean.bias",
        "yaw_out.weight": "yaw_mean.weight",
        "yaw_out.bias": "yaw_mean.bias",
    }

    loaded = 0
    for k, v in bc_state.items():
        target_k = name_map.get(k, k)
        if target_k in model_state and v.shape == model_state[target_k].shape:
            model_state[target_k] = v
            loaded += 1

    model.load_state_dict(model_state)
    print(f"Loaded {loaded}/{len(model_state)} params from BC checkpoint")


def train_ppo(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型
    encoder = Encoder3D(input_shape=(32, 32, 10), channels=ENCODER_CHANNELS, embed_dim=ENCODER_EMBED_DIM)
    model = ViewpointActorCritic(encoder, embed_dim=ENCODER_EMBED_DIM).to(DEVICE)

    if args.bc_ckpt:
        load_bc_pretrained(model, args.bc_ckpt)

    # 分组学习率
    actor_params = []
    critic_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if "critic" in name:
            critic_params.append(p)
        elif any(k in name for k in ["pos_net", "pos_mean", "yaw_net", "yaw_mean", "log_std"]):
            actor_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": actor_params, "lr": PPO_LR_ACTOR},
        {"params": critic_params, "lr": PPO_LR_CRITIC},
        {"params": backbone_params, "lr": PPO_LR_BACKBONE},
    ])

    # 环境
    env = ViewpointEnv()
    buffer = RolloutBuffer()

    # CSV 日志 + TensorBoard
    csv_path = os.path.join(args.save_dir, "ppo_log.csv")
    log_file = open(csv_path, "a", newline="")
    writer = csv.writer(log_file)
    if os.path.getsize(csv_path) == 0:
        writer.writerow(["episode", "reward", "loss"])

    start_ep = 0
    best_reward = -float("inf")
    tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))

    # 断点续训
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt.get("episode", 0) + 1
        best_reward = ckpt.get("best_reward", -float("inf"))
        print(f"Resumed from episode {start_ep}")

    # 训练循环
    timestep = 0
    for ep in range(start_ep, args.max_episodes):
        obs, _ = env.reset()
        grid = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            action, logprob, value = model.get_action(grid)
        model.train()

        action_np = action.cpu().numpy().flatten()
        obs_next, reward, terminated, truncated, info = env.step(action_np)

        buffer.add(
            grid.squeeze(0).cpu(), action.cpu().squeeze(0),
            logprob.cpu().item(), reward, value.cpu().item(), terminated,
        )
        timestep += 1

        # 进度打印
        if ep % 50 == 0:
            recent = np.mean(buffer.rewards[-50:]) if len(buffer.rewards) >= 50 else np.mean(buffer.rewards)
            errs = [1 if r < 0 else 0 for r in buffer.rewards[-50:]] if len(buffer.rewards) >= 50 else []
            err_rate = np.mean(errs) if errs else 0
            print(f"Ep {ep:5d}: r50={recent:+7.3f} err={err_rate:.0%} "
                  f"ts={timestep}/{args.update_timestep} "
                  f"std={torch.exp(model.log_std).mean().item():.3f}")

        # PPO 更新
        if timestep >= args.update_timestep:
            print(f"--- PPO UPDATE at ep={ep} ts={timestep} ---", flush=True)
            metrics = _ppo_update(model, optimizer, buffer, args)
            avg_r = np.mean(buffer.rewards)
            errors = [1 if r < 0 else 0 for r in buffer.rewards]
            err_rate = np.mean(errors)

            writer.writerow([ep, avg_r, metrics["loss"]])
            log_file.flush()
            tb_writer.add_scalar("reward", avg_r, ep)
            tb_writer.add_scalar("loss", metrics["loss"], ep)
            tb_writer.add_scalar("actor_loss", metrics["actor_loss"], ep)
            tb_writer.add_scalar("critic_loss", metrics["critic_loss"], ep)
            tb_writer.add_scalar("entropy", metrics["entropy"], ep)
            tb_writer.add_scalar("approx_kl", metrics["approx_kl"], ep)
            tb_writer.add_scalar("clip_fraction", metrics["clip_frac"], ep)
            tb_writer.add_scalar("error_rate", err_rate, ep)
            tb_writer.add_scalar("std", torch.exp(model.log_std).mean().item(), ep)

            if True:  # always print update
                kl_warn = " [KL EARLY STOP]" if metrics["kl_stopped"] else ""
                print(f"Ep {ep:5d}: reward={avg_r:+7.3f} loss={metrics['loss']:.4f} "
                      f"entropy={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f} "
                      f"err_rate={err_rate:.0%} err={info.get('error','none')}{kl_warn}")

            if avg_r > best_reward:
                best_reward = avg_r
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

            buffer.clear()
            timestep = 0

        # 保存检查点
        if ep > 0 and ep % 1000 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep,
                "best_reward": best_reward,
            }, os.path.join(args.save_dir, "checkpoint.pth"))

    log_file.close()
    tb_writer.close()
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    print(f"PPO done. Best reward: {best_reward:.3f}")


def _ppo_update(model, optimizer, buffer, args):
    grids = torch.stack(buffer.grids).to(DEVICE)
    actions = torch.stack(buffer.actions).to(DEVICE)
    old_logprobs = torch.FloatTensor(buffer.logprobs).unsqueeze(1).to(DEVICE)
    rewards = torch.FloatTensor(buffer.rewards).unsqueeze(1).to(DEVICE)
    values = torch.FloatTensor(buffer.values).unsqueeze(1).to(DEVICE)

    # 蒙特卡洛回报 (单步，reward 就是 return)
    returns = rewards
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0
    total_actor = 0
    total_critic = 0
    total_entropy = 0
    total_kl = 0
    total_clip = 0
    actual_epochs = 0
    kl_stopped = False

    for epoch_i in range(PPO_K_EPOCHS):
        logprobs, entropy, values_new = model.evaluate(grids, actions)

        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - PPO_EPS_CLIP, 1 + PPO_EPS_CLIP) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * nn.MSELoss()(values_new, returns)
        entropy_bonus = -0.1 * entropy.mean()  # stronger entropy (was 0.05)

        # KL penalty to prevent drifting too far from BC initialization
        kl_penalty = 0.01 * ((logprobs - old_logprobs).pow(2)).mean()

        loss = actor_loss + critic_loss + entropy_bonus + kl_penalty

        # KL 早停：如果策略变化太大，跳过这次更新
        with torch.no_grad():
            kl = ((ratio - 1) - (logprobs - old_logprobs)).mean().item()
        if kl > 0.05:
            if epoch_i == 0:
                kl_stopped = True
            break

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        total_actor += actor_loss.item()
        total_critic += critic_loss.item()
        total_entropy += entropy.mean().item()
        with torch.no_grad():
            clip_frac = ((ratio - 1).abs() > PPO_EPS_CLIP).float().mean().item()
        total_kl += kl
        total_clip += clip_frac
        actual_epochs += 1

    k = max(actual_epochs, 1)
    return {
        "loss": total_loss / k,
        "actor_loss": total_actor / k,
        "critic_loss": total_critic / k,
        "entropy": total_entropy / k,
        "approx_kl": total_kl / k,
        "clip_frac": total_clip / k,
        "kl_stopped": kl_stopped,
    }


def main():
    parser = argparse.ArgumentParser(description="PPO Fine-tuning")
    parser.add_argument("--bc-ckpt", type=str, default=PPO_BC_CKPT)
    parser.add_argument("--save-dir", type=str, default=PPO_SAVE_DIR)
    parser.add_argument("--max-episodes", type=int, default=PPO_MAX_EPISODES)
    parser.add_argument("--update-timestep", type=int, default=PPO_UPDATE_TIMESTEP)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train_ppo(args)


if __name__ == "__main__":
    main()
