# modal-ssh

YAML-driven CLI for Modal GPU VMs。两种模式共享同一份 yml + image:

- **`up`**: 起一台开发用的 VM,自动写 `~/.ssh/config` + 弹 VSCode Remote-SSH 窗口。适合调试、写代码、探索。
- **`run`**: 把一个 bash 脚本扔给 Modal 后台跑,本地立刻返回。适合训练、评测、批量任务。

## 两种模式对照

| | `modal-ssh up` | `modal-ssh run` |
|---|---|---|
| 容器在干什么 | sshd 阻塞等连接 | 跑你的 bash 脚本 |
| 本地副作用 | 改 `~/.ssh/config` + 开 VSCode（自动弹窗） | 只打印 log 命令提示 |
| 适合场景 | 调试、写代码、交互式探索 | 长跑训练、批量 eval、定时任务 |
| 何时停 | 执行 `down` 或 `duration_hours` 到期 | 脚本退出 / `down` / 到期 |
| 怎么看输出 | ssh 进去看终端 | `modal-ssh logs <config>` |

## Quick Start

```bash
# 一次性安装
cd modal-ssh-cli
pip install -e .

# ── 交互式 ──
modal-ssh up sglang               # 起一台 VM,首次 5-15 分钟 build
modal-ssh ssh sglang              # ssh 进去(VSCode 起完时已自动开了)
modal-ssh down sglang             # 用完停掉

# ── 后台任务 ──
modal-ssh run mrp-train-test ./train.sh    # 提交 → 本地立刻返回
modal-ssh logs mrp-train-test              # 看输出(实时 stream 或事后 replay)
modal-ssh down mrp-train-test              # 清掉

# ── 通用 ──
modal-ssh ls                      # 看所有在跑的 VM
modal-ssh configs                 # 列所有可用 yml
modal-ssh -h                      # 总 help
modal-ssh up -h                   # 子命令 help
```

---

## 安装前置

**必需**

```bash
cd modal-ssh-cli && pip install -e .   # 注册 modal-ssh 到 PATH
modal setup                            # Modal 认证(没做过的话)
```

**可选（按你的 yml 需求）**

| 前置 | 触发条件 | 怎么设置 |
|---|---|---|
| `huggingface-secret` Modal secret | yml 里 `secrets:` 引用了它(默认) | `modal secret create huggingface-secret HF_TOKEN=hf_你的token` |
| `~/.ssh/id_ed25519_modal` | 默认开启 | 第一次跑会自动 `ssh-keygen` 生成 |
| `~/.ssh/id_ed25519_github` | yml 里启用了 `github_ssh_key`(部分模板) | 生成本地 key 加到 GitHub,或注释掉 yml 里那行 |
| `code` CLI on PATH | yml 里 `open_vscode: true` 为自动开 VSCode | VSCode 里 `Cmd+Shift+P → Shell Command: Install 'code' command in PATH` |
| VSCode 平台默认 | 想避免每次连远端问平台 | `settings.json` 加 `"remote.SSH.remotePlatform": {"*": "linux"}` |

不装可选项也能跑 —— 缺什么会在对应步骤报错

---

## 命令一览

| 命令 | 干啥 |
|---|---|
| `modal-ssh up <config> [options]` | 起 dev VM,sshd 监听,顺便弹 VSCode 远程窗口 |
| `modal-ssh ssh <config> [--instance ID]` | ssh 到 `up` 起来的 VM |
| `modal-ssh down <config> [--instance ID \| --all]` | 停 VM |
| `modal-ssh ls` | 列所有 modal-ssh 起的 live app |
| `modal-ssh run <config> <script> [options]` | 提交 bash 脚本作为后台 job |
| `modal-ssh logs <config> [--instance ID]` | tail / replay job log |
| `modal-ssh configs` | 列可用 yml |

每个命令的 options 见下面分节。

---

## 命令详解

### `modal-ssh up <config> [options]`

按 `<config>` yml 起一台 dev VM。`<config>` 可以是短名(`sglang`)、文件名(`sglang.yml`)、或完整路径。省略就用 `configs/default.yml`。

| Option | 干啥 |
|---|---|
| `--gpu <spec>` | 临时覆盖 GPU,如 `B200:2` / `H100` / `A10G:1` |
| `--duration <hours>` | 临时覆盖容器寿命 |
| `--instance <id>` | 给 `app_name` 和 `job_name` 加 `-<id>` 后缀,实现同一个yml同时起多个实例 |

```bash
modal-ssh up sglang                                  # 用 yml 默认
modal-ssh up sglang --gpu A100:1 --duration 1        # 起一张 A100、跑 1 小时
modal-ssh up mrp-train-test --instance a             # 并行实例 mrp-train-test-a
```

### `modal-ssh ssh <config> [--instance ID]`

从 `~/.ssh/config` 里按 marker `modal-vm-<job_name>[-<instance>]` 找回上次 `up` 时写入的 host,然后 `exec ssh <host>`。**不会起新 VM**——VM 必须在跑。等价于本地敲 `ssh <host>`。

```bash
modal-ssh ssh sglang
modal-ssh ssh mrp-train-test --instance a
```

### `modal-ssh down <config> [--instance ID | --all]`

停掉指定 yml 的 VM

| 用法 | 行为 |
|---|---|
| `modal-ssh down sglang` | 停 `app_name == "sglang-ssh"` 的所有 live 实例(没 `--instance` 后缀那个) |
| `modal-ssh down mrp-train-test --instance b` | 只停 `mrp-train-test-b` |
| `modal-ssh down mrp-train-test --all` | 停 base + 所有 `-X` 后缀实例 |

```bash
modal-ssh down sglang
modal-ssh down mrp-train-test --instance b
modal-ssh down mrp-train-test --all
```

### `modal-ssh ls`

显示**本 CLI 起的 live app**——通过 `configs/*.yml` 声明的 `app_name` 过滤。

```
ap-XXXX...   sglang-ssh         ephemeral (detached)  running
ap-YYYY...   mrp-train-test-a   ephemeral (detached)  running
ap-ZZZZ...   mrp-train-test-b   ephemeral (detached)  zombie (no container)
```

- **running** = 容器在跑
- **zombie** = Modal 那边 app 状态还活着但容器已死。不计入费用,只是没清干净的状态记录。`down` 会清掉

### `modal-ssh run <config> <script> [options]`

把本地 `<script>` 提交为后台 job。容器跑这个脚本而不是 sshd,跑完(或挂掉)容器自然结束。

| Option | 干啥 |
|---|---|
| `--gpu <spec>` | 覆盖 GPU |
| `--duration <hours>` | 覆盖寿命 |
| `--instance <id>` | 并行实例后缀(等同 `up` 的) |
| `-f` / `--foreground` | 不 detach,本地 stream stdout |

```bash
modal-ssh run mrp-train-test ./train.sh                    # 提交,detach 返回
modal-ssh run mrp-train-test ./train.sh -f                 # 前台,本地 stream
modal-ssh run mrp-train-test ./train.sh --instance exp1
modal-ssh run mrp-train-test ./quick.sh --gpu A10G:1 --duration 1
```

脚本会**自动激活 yml build 时设的 conda 环境**(比如 `llamafactory_sdar`)。`shell_env` 里的变量(`HF_TOKEN` 等)在脚本里直接能用,**不需要**手动 `conda activate` 或 `source`。

### `modal-ssh logs <config> [--instance ID]`

底层就是 `modal app logs <app_id>`:
- 容器还在运行 → live stream(Ctrl+C 退出但不杀容器)
- 容器已经停止 → 历史 replay

Log **存在 Modal 后端**,不下载到本地。Modal 保留期到了会清,要永久存档自己 `> file.log` redirect。

```bash
modal-ssh logs mrp-train-test
modal-ssh logs mrp-train-test --instance exp1
modal-ssh logs mrp-train-test --instance exp1 > exp1.log   # 存档
```

### `modal-ssh configs`

列 `configs/*.yml`(跳过 `_` 开头),带每份的 base image:

```
  default          nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
  sglang           lmsysorg/sglang:latest
  mrp-train        nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
  mrp-train-test   nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
  mrp-eval         nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

---

## 常见任务的命令组合

### 1. 起一台 dev VM 写代码

```bash
modal-ssh up sglang                       # 等它 build 完,会自动弹 VSCode
# (在 VSCode 远程窗口里写代码、跑实验)
modal-ssh down sglang                     # 用完关掉
```

### 2. 多 ssh session 到同一台 VM

```bash
modal-ssh up sglang                       # 已经自动弹 VSCode,现在还想要终端
modal-ssh ssh sglang                      # 起一个 ssh terminal
modal-ssh ssh sglang                      # 再起一个,跟前一个互不影响
```

### 3. 跑一个一次性训练

```bash
modal-ssh run mrp-train-test ./train.sh   # 提交,走开
# 几小时后:
modal-ssh ls                              # 还在不在跑
modal-ssh logs mrp-train-test             # 看 log
modal-ssh down mrp-train-test             # 跑完了清掉 zombie 状态
```

### 4. 并行多组实验(每组独立 stoppable)

```bash
modal-ssh run mrp-train-test ./train.sh --instance lr_1e-3
modal-ssh run mrp-train-test ./train.sh --instance lr_5e-4
modal-ssh run mrp-train-test ./train.sh --instance lr_1e-4

modal-ssh ls                                          # 看 3 行
modal-ssh logs mrp-train-test --instance lr_1e-3      # 看其中一个的输出
modal-ssh down mrp-train-test --instance lr_5e-4      # 觉得这个不好,单独停
modal-ssh down mrp-train-test --all                   # 全部跑完,一次清光
```

### 5. 同一份 yml 既要 dev 又要后台跑

```bash
modal-ssh up  mrp-train-test --instance dev          # dev VM,你 ssh 上去
modal-ssh run mrp-train-test ./batch.sh --instance batch1   # 同时后台跑
modal-ssh run mrp-train-test ./batch.sh --instance batch2   # 再来一个
# 三个独立 app,互不打架
```

### 6. 用便宜的小卡先 smoke test

```bash
modal-ssh run mrp-train-test ./smoke.sh --gpu A10G:1 --duration 1 --instance smoke
modal-ssh logs mrp-train-test --instance smoke
modal-ssh down mrp-train-test --instance smoke
```


### 7. 私有 repo 怎么 clone

启用 yml 里 `github_ssh_key: ~/.ssh/id_ed25519_github`(本地这把 key 加到 GitHub 账号),然后 yml 里 `git_repo: git@github.com:org/repo.git` 会在 image build 时直接 clone 好。

---

## YAML 配置

### `_base.yml` 自动合并

`configs/_base.yml` 是**公共默认值**,起 VM 时会自动 deep-merge 到具体指定的 yml **之下**。`modal-ssh configs` 会跳过它。

合并规则:

| 字段类型 | 行为 |
|---|---|
| dict | 递归合并,key 取并集,冲突时具体 yml 赢 |
| list | **整体替换**,base 的 list 被丢弃 |
| 标量(字符串/数字/bool) | 替换 |

**list 整体替换的坑**:base 有 `apt_packages: [openssh-server, git, ...]`,你想多装 `build-essential`,**必须把整个 list 列全**:

```yaml
# 错: 这样会丢掉 base 那 6 个包
apt_packages: [build-essential]

# 对:
apt_packages: [openssh-server, git, wget, curl, tmux, vim, build-essential]
```

**dict 增量合并的便利**:base 里 `volumes: {/root/.cache/huggingface: huggingface-cache}`,你只加新的就行:

```yaml
volumes:
  /root/checkpoints: my-ckpts        # 会和 base 的 HF cache 合并,两个都生效
```

### 支持的 YAML 字段

| 字段 | 说明 |
|---|---|
| `app_name` | Modal 那边显示的 app 名,`down` / `ls` / `logs` 都按这个匹配 |
| `job_name` | `~/.ssh/config` 里 marker 名 `modal-vm-<job_name>`,`ssh` 用它定位 host |
| `open_vscode` | bool,起完是否自动开 VSCode Remote-SSH |
| `base_image` | 容器基础镜像 |
| `add_python` | Modal 要给 base image 加上的 Python 版本(base 自带 Python 就省略) |
| `gpu.type` / `gpu.count` | 例如 `H100` / `4` |
| `duration_hours` | 容器寿命,到期自动停 |
| `ssh_public_keys` | 注入 `authorized_keys` 的本地 pubkey 列表(第一个的私钥用作默认 IdentityFile) |
| `github_ssh_key` | 可选,本地 GitHub 私钥路径,烤进镜像让容器里能 git clone 私有 repo |
| `git_repo` | 可选,build 时 `git clone` 的 repo,字符串或 `{url, dest}` |
| `apt_packages` | build 时 `apt install` 的包 |
| `run_commands` | build 时跑的额外 shell 命令(每条一个 image layer) |
| `volumes` | `mount_path: volume_name` 字典,挂 Modal Volume,持久化 |
| `secrets` | Modal Secret 名字列表 |
| `shell_env` | export 到 `/root/.profile` + `/root/.bashrc` 的 env vars。值里 `$VAR` 引用容器侧 env(比如 secret 注入的 `$HF_TOKEN`) |
| `local_files` | 可选,`local_path: remote_path` 字典,把本地任意文件烤进镜像 |
| `auto_generate_modal_key` | bool,默认 true。`~/.ssh/id_ed25519_modal` 不存在就自动 `ssh-keygen` |

### 现有 configs

| 文件 | 用途 |
|---|---|
| `_base.yml` | 公共默认值,被合并,不是入口 |
| `default.yml` | 最简 GPU VM,ubuntu + cuda + Python,没有任何额外环境 |
| `sglang.yml` | sglang 开发环境(`lmsysorg/sglang:latest` base + conda + sglang) |
| `mrp-train.yml` | multi-token-denoising 训练环境(conda + torch cu128 + flash-attn + 私有 repo) |
| `mrp-eval.yml` | opencompass 评测环境(conda + torch + flash-attn + vllm + opencompass) |
| `mrp-train-test.yml` | 隔离的个人测试环境,仿 mrp-train 但 `app_name`/`job_name` 都加 `-test` 后缀 |

---

## 并行多实例(`--instance`)

同一份 yml 起多份时,默认 `down` 会**一锅端**(因为按 app_name 匹配)。要保留独立 stop 能力,用 `--instance`:

```bash
modal-ssh up  mrp-train-test --instance a     # app_name 变成 mrp-train-test-a
modal-ssh run mrp-train-test ./t.sh --instance b   # 同样的 yml,后台 job,变成 mrp-train-test-b

modal-ssh ls                                  # 两行
modal-ssh down mrp-train-test --instance a    # 只停 a,b 继续
modal-ssh down mrp-train-test --all           # 一锅端 base + 所有 -X
```

`--instance` 同时给 `app_name` 和 `job_name` 加后缀,所以:

- Modal 那边 N 个 app 独立,`down` / `ls` / `logs` 能精确区分
- `~/.ssh/config` 写 N 段独立 marker `modal-vm-<job>-<id>`,互不覆盖
- 共享同一个 image cache(后缀不烤进 image env),build 一次之后切实例秒过

**volume 共享的注意点**: `volumes:` 写死在 yml 里,所有 instance **共享同一个 volume**。HF cache 这种读多写少没事;**checkpoint 不要往同一个卷写**——要么每个实验复刻 yml,要么训练脚本里用不同子目录(`/root/checkpoints/run-a/`、`/root/checkpoints/run-b/`)。

---

## 仓库结构

```
modal-ssh-cli/
├── modal_ssh.py            # Modal app + 容器 sshd + local launcher 三合一
├── modal_ssh_cli.py        # CLI 入口(up/down/ls/ssh/run/logs/configs)
├── pyproject.toml          # console_scripts 注册 modal-ssh
├── README.md
├── configs/
│   ├── _base.yml           # 公共默认值,自动合并
│   ├── default.yml
│   ├── sglang.yml
│   ├── mrp-train.yml
│   ├── mrp-eval.yml
│   └── mrp-train-test.yml
└── scripts/
    └── download_models.sh  # VM 内部跑的工具,跟启动栈无关
```

