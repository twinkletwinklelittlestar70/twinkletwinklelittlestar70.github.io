# twinkletwinklelittlestar70.github.io
一个hexo博客项目

## Git Address
https://github.com/twinkletwinklelittlestar70/twinkletwinklelittlestar70.github.io

## 新建文章
在`/source/_posts/`下创建md文件，并完成文章写作。

## 本地调试
```
/** 本地调试 */
npm run build
/** 然后打开 http://localhost:4000 看效果 */
```

## Publish
```
/** 发布到git page */
git push
npm run deploy
/** 然后打开 https://twinkletwinklelittlestar70.github.io/ 看效果 */
/** 注意可能有缓存，多刷几遍 */
```

## 分支
项目使用hexo，默认的main分支是构建结果，不是源代码。

源代码应当推到develop分支上，然后运行发布命令。

## Hexo命令
```
/** Hexo 命令 */
/** 构建 */
hexo g

/** 本地服务 */
hexo s

/** 发布到git page */
hexo d
```

## 更换Themes
主题主页：https://github.com/fluid-dev/hexo-theme-fluid
配置：https://hexo.fluid-dev.com/docs/guide/#slogan-%E6%89%93%E5%AD%97%E6%9C%BA

