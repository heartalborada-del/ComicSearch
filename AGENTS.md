# ComicSearch 项目指南

## 项目概述

ComicSearch 是一个基于 FastAPI 的漫画/图像搜索后端，使用 SQLAlchemy ORM、ONNXRuntime 图像嵌入和 Qdrant 向量检索。前端为新增模块，需与现有后端 API（`/search`、`/info`、`/ehentai/import/tasks`、`/tasks` 等）对接。

---

## 前端架构要求

### 技术栈

- **框架**: Vue 3（Composition API，`<script setup>` 语法）
- **UI 库**: Vuetify 3
- **构建工具**: Vite
- **语言**: TypeScript
- **状态管理**: Pinia
- **路由**: Vue Router

### Material You Design 规范

前端须遵循 Material You（Material Design 3）设计语言：

- **动态配色**: 使用 Vuetify 的 theme system 配置 M3 色板（primary、secondary、tertiary、surface、background、error），支持亮色/暗色主题切换
- **圆角与形状**: 采用 M3 的大圆角风格（卡片 `rounded-lg` / `rounded-xl`，按钮 `rounded-pill` 或 `rounded-lg`）
- **层级与阴影**: 使用 M3 的 tonal elevation 概念，卡片使用低阴影 + surface 色调区分层级
- **排版**: 使用 M3 type scale（display、headline、title、body、label），通过 Vuetify `typography` 配置
- **组件选用**: 优先使用 Vuetify 的 M3 风格组件（`v-card`、`v-list`、`v-navigation-drawer`、`v-app-bar`、`v-chip`、`v-btn` variant `tonal` / `outlined` / `text`）
- **动效**: 使用 Vuetify 内置 transition，页面切换和组件交互须有平滑过渡

### 多端适配要求

- **响应式布局**: 使用 Vuetify 的 `v-col` / `v-row` 栅格系统（12 列），配合 `display` 断点（`xs` `sm` `md` `lg` `xl`）实现响应式
- **移动端优先**: 默认以移动端视图设计，再向上适配平板和桌面
- **导航适配**:
  - 移动端：底部导航栏（`v-bottom-navigation`）或抽屉式导航（`v-navigation-drawer`）
  - 桌面端：侧边栏导航（`v-navigation-drawer` rail 模式可折叠）
- **触摸友好**: 可交互元素最小点击区域 44×44px，移动端禁用 hover 依赖逻辑
- **图片自适应**: 搜索结果图片使用 `v-img` 配合 `aspect-ratio` 和 `cover` 模式，避免布局偏移
- **断点行为**: 关键布局在 `sm`（600px）和 `md`（960px）断点处须有明确的布局切换

### 与后端对接

- API 基地址通过环境变量 `VITE_API_BASE_URL` 配置，默认 `http://localhost:8000`
- 所有 API 调用封装在 `src/api/` 目录下，按模块组织（`search.ts`、`tasks.ts`、`info.ts`）
- 使用 `axios` 或 `fetch` 封装统一请求层，处理错误和加载状态
- 上传图片搜索时使用 `multipart/form-data`，注意 10MB 大小限制和允许的类型（`image/jpeg`、`image/png`、`image/webp`）

---

## 代码风格要求

### 通用规则

- **缩进**: 2 个空格（前端 TypeScript / Vue 文件），4 个空格（后端 Python 文件）
- **行宽**: 前端 100 字符，后端 120 字符
- **引号**: 前端使用单引号 `'`，后端使用双引号 `"`
- **分号**: 前端语句末尾不加分号（ESLint `semi: false`）
- **换行符**: LF（`\n`），文件末尾保留一个空行
- **编码**: UTF-8，无 BOM

### TypeScript / Vue 规则

- 始终使用 `<script setup lang="ts">`，不使用 Options API
- 所有函数参数和返回值须有显式类型标注，禁止使用 `any`（确需时使用 `unknown` 并收窄）
- 接口和类型定义放在 `src/types/` 目录，按领域分文件
- 组件名使用 PascalCase（`ComicCard.vue`），组件文件名与组件名一致
- Props 使用 `defineProps<T>()` 泛型语法定义类型
- Emits 使用 `defineEmits<T>()` 泛型语法定义类型
- 组合式函数（composables）放在 `src/composables/`，命名以 `use` 开头（`useSearch.ts`）
- 常量使用 `UPPER_SNAKE_CASE`，变量和函数使用 `camelCase`
- 避免使用 `var`，使用 `const` 或 `let`
- 模板中组件使用 PascalCase 标签（`<ComicCard />` 而非 `<comic-card />`）

### Python 规则（后端）

- 文件顶部始终添加 `from __future__ import annotations`
- 类型标注使用 `|` 联合语法（`str | None`），不使用 `Optional[str]`
- 导入顺序：标准库 → 第三方库 → 本地模块，各组之间空一行
- 类和函数使用三引号 docstring 描述用途
- 模块级常量使用 `UPPER_SNAKE_CASE`
- 私有成员以单下划线前缀（`_internal_var`）

### 命名约定

| 对象 | 前端 | 后端 |
|------|------|------|
| 变量/函数 | `camelCase` | `snake_case` |
| 常量 | `UPPER_SNAKE_CASE` | `UPPER_SNAKE_CASE` |
| 类/组件 | `PascalCase` | `PascalCase` |
| CSS 类 | `kebab-case`（BEM 可选） | — |
| 文件名（组件） | `PascalCase.vue` | — |
| 文件名（其他） | `kebab-case.ts` | `snake_case.py` |

### 目录结构（前端）

```
frontend/
├── public/
├── src/
│   ├── api/            # API 请求封装
│   ├── assets/         # 静态资源
│   ├── components/     # 通用组件
│   ├── composables/    # 组合式函数
│   ├── layouts/        # 布局组件
│   ├── pages/          # 页面级组件（views）
│   ├── router/         # 路由配置
│   ├── stores/         # Pinia store
│   ├── styles/         # 全局样式与主题配置
│   ├── types/          # TypeScript 类型定义
│   └── App.vue
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

### Lint 与格式化

- 前端: ESLint + Prettier，保存时自动格式化
- 后端: Ruff（lint + format）
- 提交前须通过 lint 检查，无 error 级别问题

---

## 构建与测试

### 前端

```bash
cd frontend
npm install
npm run dev       # 开发服务器
npm run build     # 生产构建
npm run preview   # 预览构建产物
npm run lint      # ESLint 检查
npm run type-check  # TypeScript 类型检查
```

### 后端

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest tests/
```

---

## 约定

- 前端代码和后端代码分目录存放，前端放在 `frontend/` 目录下，不污染后端 `app/` 结构
- 新增 API 端点后须同步更新前端 API 封装和类型定义
- 搜索结果页面须支持分页/懒加载，避免一次性渲染大量图片
- 异步任务（ehentai 导入）须有明确的状态展示（pending / running / success / failed）和取消入口
- 暗色主题须完整支持，所有自定义样式使用 CSS 变量或 Vuetify theme 变量，不硬编码颜色值
