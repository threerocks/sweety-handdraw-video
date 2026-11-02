# 发布一个稳定版本

`skills/sweety-handdraw-video/VERSION` 是版本源。版本已公开后不移动原标签；修复发布使用递增版本。

1. 修改 VERSION、首页当前版本和“需要固定本版时”的 ZIP 链接、包内 README 当前版本；新增 `docs/releases/v<版本>.md`。新版安装包尚未发布时，这些变更应与标签一起发布。
2. 运行 `python3 tools/release.py check`。该命令只核对本地版本与入口，不声称远程附件已经存在。
3. 运行 README 中的回归和离线冒烟检查，提交变更，在干净的该提交创建匹配的注释标签。提交说明用 `Closes #编号` 列出本版已完成的 bug 和子 issue（见下文），合入默认分支后会自动关闭。
4. 将代码与指定标签推送。`Publish tagged skill` 会等待 macOS / Linux 的完整 Verify 工作流通过，再从标签所指提交打包。它检查 ZIP 内 VERSION、整包和逐文件哈希后，上传 ZIP、`SHA256SUMS.txt`、`package-sha256.json`，最后公开 Release。
5. 回读远程提交、标签与 Release 附件，下载 ZIP 和 SHA256SUMS.txt 实测校验。解压到空目录，读取 VERSION 并运行 `scripts/validate_package.py`。用户已有安装副本不会因仓库更新而自动替换。

发布工作流支持手动输入**已有标签**重跑。已有公开 Release 时只验证附件哈希，不覆盖其文件；草稿允许重新上传同一标签的构建结果。远程 CI 未完成、失败或安装包未能实际下载时，应如实报告状态。

## 用 Issue 跟踪一个版本

- **缺陷**：每个缺陷单独开一个 issue，打上 `bug` 标签。写清现象、影响和修复方式；有实测数字就写进去。
- **其余改进**：本版只开一个总 issue（例如 3.0.0 的 [#19](https://github.com/threerocks/sweety-handdraw-video/issues/19)），所有改进都挂在它下面，作为 GitHub 子 issue，打上 `enhancement` 标签。
  - 按阶段或互不相干的目标拆，通常 5 个左右，不拆成细碎任务。
  - 总 issue 正文里同时列出本版修复的 bug 编号，方便一眼看清整版的范围。
- **状态**：在本地修好、还没提交的，在 issue 里写明“已在本地完成，随 <版本> 提交”，不要提前关闭。
  - 发布提交合入默认分支后，相应 issue 会自动关闭。
  - 子 issue 全部关闭后，手动关闭总 issue。
  - 本版没做完的，移到下一版的总 issue 下。
- **公开内容**：仓库是公开的。issue 里不写本机路径，也不写未公开的内容。
