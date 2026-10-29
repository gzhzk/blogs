# Zekai · Notes

个人博客：[gzhzk.github.io/blogs/](https://gzhzk.github.io/blogs/)。基于 Astro + [Fuwari](https://github.com/saicaca/fuwari)。

个人主页独立维护于 [gzhzk/gzhzk.github.io](https://github.com/gzhzk/gzhzk.github.io)。本仓库仅发布 `/blogs/`，不会覆盖主页。

## 本地写作

需要 Node.js 22+ 和 pnpm 9.14.4。

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

打开终端提示的 `/blogs/` 地址。文章放在 `src/content/posts/`；图片放在 `public/images/`，正文中使用 `![说明](/blogs/images/文件名.png)`。

```sh
pnpm new-post my-note
```

新文章默认 `draft: true`，准备好后改为 `false`。`published` 为发布日期，`updated` 为修改日期。数学公式用 `$...$` 或独立成行的 `$$...$$`；代码用 Markdown 围栏并标明语言（如 `python`）。不需要编写排版 HTML。

注意：这是公开源码仓库，`draft: true` 仅控制页面生成。私密草稿先保存在本地 `.local-drafts/`，不要提交。

## 发布

```sh
pnpm check
pnpm build
git add .
git commit -m "Update notes"
git push
```

推送到 `main` 后，GitHub Actions 自动检查、构建、生成 Pagefind 搜索索引并发布到 GitHub Pages。无需 Hexo，也不需要手动上传 `dist/`。Pages 的 Source 配置为 GitHub Actions。

## 内容与主题

三个栏目通过文章 front matter 的 `category` 归类：`前沿模型`、`算法与组件`、`开源项目拆解`。原有个人随笔保留为 `随笔`。栏目介绍与选题方向在 `src/data/columns.ts`；首页和栏目页自动收集对应文章。选题标签不代表已发布文章。

前沿模型目标每周一篇；写作提纲见 [WRITING.md](WRITING.md)。

- `src/content/posts/`：迁移的三篇正式文章，保留原日期、标签、分类、代码和公式。
- `src/config.ts`：站点名称、简介、导航、蓝色主题、头像。
- `astro.config.mjs`：`site: https://gzhzk.github.io`，`base: /blogs/`。
- `public/images/`：文章配图。
- `migration-report.json`：迁移对应关系和完整性核对。

原 Hexo 的内联排版 HTML 已转为 Markdown，文章观点和示例代码未做技术修订。原稿中的空白章节保持原样。

未发布的 Hexo 文稿保存在本地迁移备份中，未上传此仓库。旧 Hexo 目录停止用于发布，尤其不要再运行指向主页仓库的 `hexo deploy`。

## 许可

主题代码保留上游 Fuwari 的 MIT 许可（见 `LICENSE`）；文章与个人图片的权利归各自作者所有，不因主题许可而授予文章内容许可。
