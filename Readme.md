![LS](https://cdn.modrinth.com/data/5vV6tCDu/images/0ab34a46d9814be15d35213f5963814f865ed50d.png)

# 在 compute.toys 上手写一个 Minecraft：原理拆解

> 对象：单文件 WGSL compute shader，64×16×64 体素世界，第一人称行走，可挖掘/放置。
> 环境：compute.toys（WebGPU 计算着色器游乐场）。

---

## 0. 整体架构

一句话：**把"每个像素求一条射线的命中点"写成一个线程的任务，塞进 compute dispatch 里跑。**

```
CPU / 宿主
 ├─ screen       输出存储纹理（画布）
 ├─ channel0/1   两张方块贴图
 ├─ mouse/time   prelude 提供的输入
 └─ world        #storage 声明的持久 storage buffer
        │
        ▼
一次 dispatch，每个像素一个线程：
   1. 从 world 读玩家位置
   2. 读键盘/鼠标，推进物理（只由 (0,0) 线程写回）
   3. 从眼睛发一条射线，DDA 步进找到命中的体素
   4. 按命中面采样 channel0/1，加面光照
   5. textureStore 到 screen
```

它其实是一个**被改造成 compute 的延迟渲染器**：没有顶点缓冲、没有光栅化，几何完全由一条射线和一张体素表实时求交得到。

### 用到的宿主资源

| 名字 | 类型 | 用途 |
|---|---|---|
| `screen` | `texture_storage_2d` | 唯一输出 |
| `channel0` / `channel1` | `texture_2d<f32>` | 土 / 石贴图 |
| `bilinear` | `sampler` | 采样器 |
| `mouse` | struct | `pos`（像素坐标）、`click`（0/1/3）、`delta`、`zoom` |
| `time` | struct | `elapsed`、`delta`、`frame` |
| `keyDown(k)` | 函数 | 键盘状态查询 |
| `SCREEN_WIDTH/HEIGHT` | 常量 | 画布尺寸 |

`#storage world array<u32>` 是 compute.toys 的预处理器指令，展开成一条 `@group @binding var<storage, read_write> world: array<u32>;`，由工具自动分配并绑定。

---

## 1. 世界怎么存

### 1.1 一个 `u32` 一个方块

```wgsl
#storage world array<u32>
```

- 方块 id：`0` = 空气，`1` = 土，`2` = 石
- 世界 64 × 16 × 64 = **65536** 个格子，正好 256 KB

索引用行主序，`y` 在最外层：

```wgsl
fn idxOf(p: vec3i) -> u32 {
    return u32(p.x) + u32(p.z) * u32(WX) + u32(p.y) * u32(WX * WZ);
}
// 即 idx = x + z*64 + y*4096
```

这样同一个 `y` 层的 64×64 是一个连续区间，**垂直方向的邻居索引差恒为 4096**，后续如果想做列式优化（比如一次读一整列）会很方便。

不用结构体、不用 3D 纹理，只因为：`u32` 最省，随机访问没有采样器语义干扰，也不用处理格式转换。

### 1.2 关键设计：程序化地形 + 编辑增量

这是整个实现里最省事的一招。

```wgsl
fn getBlock(p: vec3i) -> u32 {
    if (!inBounds(p)) { return 0u; }

    let e = world[idxOf(p)];
    if (e != 0u) { return e - 1u; }      // ① 被编辑过，直接返回

    let h = heightAt(p.x, p.z);          // ② 否则现算程序化地形
    if (p.y < h)  { return 2u; }
    if (p.y == h) { return 1u; }
    return 0u;
}
```

约定：**`world` 里存 `v + 1`，`0` 专门表示"这一格从没被改过"**。

于是：

| 情况 | world 里的值 | 读出来 |
|---|---|---|
| 没改过 | `0` | 走 `heightAt` 现算 |
| 改成空气 | `1` | 空气 |
| 改成土 | `2` | 土 |
| 改成石 | `3` | 石 |

**为什么这么设计：**

1. **不需要初始化 buffer。** WebGPU 创建的 buffer 一定是零初始化的，而 `0` 恰好就是"未改动"，所以第一次运行就直接是完整地形，不需要任何 init pass 或 CPU 上传。
2. **buffer 大小和地形无关。** 即使 buffer 分配得很小，前半部分地形照样存在；只有超出范围的编辑会写不进去（WebGPU 对越界写入是静默丢弃）。
3. **只存改动。** 挖几块就占几个 u32，其余全是 0。

代价是每次读方块多一次分支和几次三角函数。`heightAt` 里是 3 组 `sin/cos`，在 DDA 内层循环里每步都要调一次，这是这个实现最大的性能开销来源——后面会讲怎么优化。

### 1.3 状态槽：把浮点数塞进 u32 数组

```wgsl
const STATE : u32 = 65536u;
// 65536 = pos.x, 65537 = pos.y, 65538 = pos.z, 65539 = 上一帧帧号
```

`array<u32>` 存不了 `f32`，所以用位重解释：

```wgsl
fn loadPos() -> vec3f {
    return vec3f(bitcast<f32>(world[STATE]),
                 bitcast<f32>(world[STATE + 1u]),
                 bitcast<f32>(world[STATE + 2u]));
}

fn storePos(p: vec3f, f: u32) {
    world[STATE]      = bitcast<u32>(p.x);
    world[STATE + 1u] = bitcast<u32>(p.y);
    world[STATE + 2u] = bitcast<u32>(p.z);
    world[STATE + 3u] = f;
}
```

注意 `bitcast` 不是数值转换：`bitcast<u32>(1.0)` 得到的是 `0x3F800000`，不是 `1`。这里要的就是位重解释。

---

## 2. 地形生成

```wgsl
fn heightAt(x: i32, z: i32) -> i32 {
    let fx = f32(x);
    let fz = f32(z);
    let hf = 5.0 + 2.5 * sin(fx * 0.18) * cos(fz * 0.15)
                 + 1.8 * sin(fx * 0.07 + fz * 0.11)
                 + 0.6 * sin(fx * 0.53) * sin(fz * 0.47);
    return i32(clamp(hf, 0.0, f32(WY) - 1.0));
}
```

三个频率叠加：

| 项 | 频率 | 作用 |
|---|---|---|
| `sin(0.18x)·cos(0.15z)` | 低 | 大尺度丘陵 |
| `sin(0.07x + 0.11z)` | 更低 | 整体倾斜、形成盆地和山脊 |
| `sin(0.53x)·sin(0.47z)` | 高 | 细碎起伏，避免表面太光滑 |

用 `sin/cos` 的组合而不是真正的噪声（Perlin/Simplex），是因为它**无限可微、无平铺感、零查表**，几行就够。缺点是会出现明显的正弦纹理，但对一个玩具来说够用。

`clamp` 到 `[0, WY-1]` 保证不会越界。实际值域约 `0.1 ~ 9.9`，所以世界里最高的山在 `y ≈ 9`，还有 6 格空间可以往上盖。

分层很直接：

```wgsl
if (p.y < h)  { return 2u; }   // 石头
if (p.y == h) { return 1u; }   // 土（只有最上面一层）
return 0u;                      // 空气
```

**为什么用 2D 高度场而不是 3D 噪声：**

- 保证不会出现悬浮方块（3D 噪声会），玩家不会莫名其妙卡在半空
- 保证每一列都是实心的，`groundAt` 只需要从上往下找第一个非空，不用处理"头顶有洞"
- 便宜：只需 2 次 `f32` 转换 + 3 组三角函数，无循环

---

## 3. 相机

### 3.1 这一版的做法：鼠标绝对位置映射

```wgsl
var mpos = vec2f(mouse.pos);
if (mpos.x <= 0.0 && mpos.y <= 0.0) { mpos = res * 0.5; }

let yaw   = (mpos.x / res.x - 0.5) * 4.0;    // ±2.0 弧度 ≈ ±114°
let pitch = -(mpos.y / res.y - 0.5) * 1.4;   // ±0.7 弧度 ≈ ±40°
```

把鼠标在屏幕上的归一化位置直接当成角度：

- 鼠标在屏幕左边缘 → `yaw = -2.0`（朝左）
- 鼠标在屏幕中心 → `yaw = 0`（朝前）
- 鼠标在右边缘 → `yaw = +2.0`（朝右）

`pitch` 前面的负号是因为屏幕 y 向下、而抬头应该是 pitch 增大。

**这个方案的优点**：不需要指针锁定，不需要累加状态，天然可逆（鼠标在哪，视角就在哪），对玩具来说很直观。

**致命缺点**：视角和鼠标位置强绑定。你想瞄准屏幕左边的东西，就必须把鼠标移过去，而这一移视角就转了——**瞄准和转视角是同一个动作**，和"左键挖掘"抢同一套输入。这就是后来改用 `mouse.delta` 累加的原因。

### 3.2 从角度到基向量

```wgsl
let fw = vec3f(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch));

let rt = normalize(cross(fw, vec3f(0.0, 1.0, 0.0)));  // 屏幕右
let up = cross(rt, fw);                               // 屏幕上
```

`fw` 是单位向量（`sin²+cos²` 恒为 1），`yaw = 0, pitch = 0` 时 `fw = (0,0,1)`，即朝 `+z`。

`rt = cross(fw, (0,1,0))` 的含义：**前方向叉乘世界上方向，得到右方向**。这是右手坐标系的标准构造，而且因为叉乘掉了 `(0,1,0)` 分量，`rt` 永远水平——抬头低头时画面不会歪（没有 roll）。

`up = cross(rt, fw)` 再叉回来，得到既垂直于 `fw` 又垂直于 `rt` 的方向，保证三个轴正交。

### 3.3 视线射线

```wgsl
let uv = screenUV(vec2f(id.xy), res);
let rd = normalize(uv.x * rt + uv.y * up + FOV_K * fw);
```

这就是经典的**针孔相机模型**：屏幕上的每个点，方向 = 右偏移 × 右向量 + 上偏移 × 上向量 + 焦距 × 前向量。

`FOV_K = 1.0` 是"焦距"。它和视野的关系：

```
垂直半视角 = atan(0.5 / FOV_K)  ≈ 26.6°
水平半视角 = atan(0.5 * aspect / FOV_K) ≈ 41.5°（16:9）
```

`FOV_K` 越大，视野越窄（长焦）；越小，视野越广。

### 3.4 上下颠倒的修复

```wgsl
fn screenUV(px: vec2f, size: vec2f) -> vec2f {
    let n = px / size - vec2f(0.5);
    return vec2f(n.x * (size.x / size.y), -n.y);   // ← 负号是关键
}
```

compute shader 里 `id.y = 0` 是**图像顶部**（和纹理行 0 一致），但相机的 `up` 方向需要"屏幕上方 = +up"。如果不取反，屏幕顶部拿到的是 `uv.y = -0.5`，射线朝下——整幅画面上下翻转。

同一条公式里的 `n.x * aspect` 负责**宽高比校正**，否则在非正方形画布上画面会被横向拉伸。

---

## 4. 体素射线步进（DDA）

这是整个渲染器的核心算法。

### 4.1 为什么不能固定步长

最朴素的做法是沿射线每隔 `ε` 采样一次：

```
for t = 0; t < maxDist; t += 0.01 { if (solid(p + rd*t)) return hit; }
```

问题：步长太小则步数爆炸（穿过 64 格世界要 6400 步），步长太大则**会穿过薄方块**——1 格厚的墙在 0.5 的步长下有一半概率被跳过。

体素世界需要的是"**每次精确跳到下一个格子边界**"，这就是 Amanatides & Woo 的 DDA 算法。

### 4.2 先用 AABB 裁剪

```wgsl
let inv = 1.0 / rd;
let ta  = (vec3f(0.0, 0.0, 0.0) - ro) * inv;
let tb  = (vec3f(f32(WX), f32(WY), f32(WZ)) - ro) * inv;
let lo  = min(ta, tb);
let hi  = max(ta, tb);
let tn  = max(max(lo.x, lo.y), lo.z);   // 进入包围盒的 t
let tf  = min(min(hi.x, hi.y), hi.z);   // 离开包围盒的 t
if (tf < max(tn, 0.0)) { return h; }    // 不相交
```

`1.0 / rd` 把"求交"变成对 slab 的除法；`min/max` 处理方向随符号翻转。取三个轴的交集：**最大的进入值**和**最小的离开值**。

这一步有三个作用：

1. **快速剔除**：相机看天空时直接返回，不进循环
2. **避免起点越界**：相机在世界外面（比如 `y = 14`，而世界只有 `0..15`）时，如果直接从 `floor(ro)` 开始，第一个格子就在世界外，`inBounds` 会立刻 `break`
3. **跳过空白**：从 `max(tn, 0)` 开始，等于跳过了从相机到世界边界之间那段无意义的距离

### 4.3 三个轴的"下一次跨格"

```wgsl
var t  = max(tn, 0.0) + 1e-4;
let p0 = ro + rd * t;
var c  = vec3i(floor(p0));       // 起始格子
```

然后对每个轴独立计算：

```wgsl
if (abs(rd.x) > 1e-8) {
    sx = i32(sign(rd.x));                                 // 步进方向 ±1
    dx = abs(1.0 / rd.x);                                  // 跨一格的 t 增量
    tx = t + (f32(c.x) + step(0.0, rd.x) - p0.x) / rd.x;   // 下一个 x 边界的 t
}
```

三者的含义：

| 变量 | 含义 |
|---|---|
| `sx` | 沿 x 轴每次走 `+1` 还是 `-1` |
| `dx` | 沿 x 轴走一格需要多少 `t`（等于 `|1/rd.x|`，因为沿轴前进 `1` 单位对应参数 `1/rd.x`） |
| `tx` | 从当前点走到下一个 x 平面边界的**绝对** `t` |

`step(0.0, rd.x)` 是个小技巧：`rd.x >= 0` 返回 1，否则返回 0。所以正方向找 `c.x + 1` 这个边界，负方向找 `c.x`。等价于 `select(0.0, 1.0, rd.x > 0.0)`，但少一次分支。

`abs(rd.x) > 1e-8` 的判断用来处理**射线平行于某个轴**的情况：此时 `dx/tx` 保持 `1e30`，那个轴永远不会被选中，等价于"这个方向不前进"。

### 4.4 主循环

```wgsl
var n = vec3i(0, 0, 0);
for (var k = 0; k < 256; k++) {
    if (!inBounds(c)) { break; }
    if (getBlock(c) != 0u) {
        h.hit = true; h.cell = c; h.nrm = n; h.t = t;
        return h;
    }
    if (tx < ty && tx < tz) {
        t = tx; c.x += sx; tx += dx; n = vec3i(-sx, 0, 0);
    } else if (ty < tz) {
        t = ty; c.y += sy; ty += dy; n = vec3i(0, -sy, 0);
    } else {
        t = tz; c.z += sz; tz += dz; n = vec3i(0, 0, -sz);
    }
}
```

每一步：

1. 检查当前格子；实心则命中返回
2. 否则选三个 `tx/ty/tz` 里**最小**的那个轴推进——这保证按空间顺序穿过格子，一次不漏
3. 推进后该轴的 `t` 增加 `d`（下一次边界的间隔恒定）

**法线的更新**：`n = vec3i(-sx, 0, 0)`。从 `-x` 方向进入一个格子意味着我们穿过了它的 `+x` 面，所以法线是 `+x`；`-sx` 正好表达"与前进方向相反"。这个法线后面同时用于**光照、贴图 uv 选择、放置位置**。

**步数上限 256**：世界最大对角 = `64 + 16 + 64 = 144` 格，256 足够覆盖任何穿越路径。设太大浪费编译/寄存器，设太小会截断长射线导致远处出现空洞。

**为什么不检查 `t` 的上限**：因为 `inBounds` 会在射线离开世界时 `break`，天然终止。

---

## 5. 贴图与光照

### 5.1 命中面的 uv

```wgsl
let p = pos + rd * h.t;          // 命中点（世界坐标）
let f = p - vec3f(h.cell);       // 格内坐标，分量在 0..1

var tuv: vec2f;
if (h.nrm.x != 0)      { tuv = vec2f(f.z, 1.0 - f.y); }  // 左右面
else if (h.nrm.y != 0) { tuv = vec2f(f.x, f.z); }         // 上下面
else                   { tuv = vec2f(f.x, 1.0 - f.y); }   // 前后面
```

原理：**贴在一个面上时，只取垂直于法线的两个分量**。

- 法线是 x 轴 → 面在 `y-z` 平面 → 用 `(f.z, f.y)`
- 法线是 y 轴 → 面在 `x-z` 平面 → 用 `(f.x, f.z)`
- 法线是 z 轴 → 面在 `x-y` 平面 → 用 `(f.x, f.y)`

`f.y` 取反是为了让贴图在竖直面上方向一致（否则顶面和侧面的图案会上下翻转）。

`f` 之所以可靠，是因为 `p` 是**精确的边界交点**（DDA 的 `t` 就是边界参数），所以 `f` 的两个切向分量必定落在 `[0,1]`，不会因为浮点误差采样到格子外。

### 5.2 材质选择

```wgsl
var tex: vec3f;
if (getBlock(h.cell) == 2u) {
    tex = textureSampleLevel(channel1, bilinear, tuv, 0.0).rgb;   // 石
} else {
    tex = textureSampleLevel(channel0, bilinear, tuv, 0.0).rgb;   // 土
}
```

按方块 id 二选一。因为是 compute stage，**只能用 `textureSampleLevel` 并显式给 LOD**——`textureSample` 依赖屏幕空间导数，compute 里没有这个概念。

LOD 固定为 0，因为：
- 方块贴图是像素画，mip 会让它糊掉
- 图集/重复平铺的贴图用 mip 会产生跨格渗色

### 5.3 假方向光

```wgsl
var shade = 0.80;
if (h.nrm.y != 0)      { shade = select(0.5, 1.0, h.nrm.y > 0); }  // 顶 1.0 / 底 0.5
else if (h.nrm.x != 0) { shade = 0.70; }                            // 左右面 0.7
// 前后面保持 0.8
```

只用**面法线**给一个常数亮度：

| 面 | 亮度 | 视觉含义 |
|---|---|---|
| 顶面 | 1.00 | 正对天光 |
| 前后面（z） | 0.80 | 侧光 |
| 左右面（x） | 0.70 | 侧光，稍暗以区分方向 |
| 底面 | 0.50 | 背光 |

这不是物理光照，但它给了三个关键信息：**面朝向可辨、体积感、相邻同材质方块之间有明暗分界**。成本是一个比较 + 一次 `select`，几乎没有。

真正的光照（法线点乘光方向 + 环境光 + 阴影）在这里性价比很低，因为体素世界的"细节"全靠贴图和棱角，明暗只要区分朝向就够了。

### 5.4 天空

```wgsl
col = mix(vec3f(0.55, 0.72, 0.95), vec3f(0.15, 0.38, 0.80),
          clamp(rd.y * 1.5 + 0.5, 0.0, 1.0));
```

按射线的 `y` 分量在地平线色和天顶色之间插值。`rd.y * 1.5 + 0.5` 把 `[-1,1]` 映射到 `[-1,2]` 再 `clamp` 到 `[0,1]`，等于让地平线以下快速收敛到地平线色。

---

## 6. 玩家与物理

### 6.1 地面吸附

```wgsl
fn groundAt(x: f32, z: f32) -> f32 {
    let bx = i32(floor(x));
    let bz = i32(floor(z));
    var top = 0;
    for (var y = WY - 1; y >= 0; y = y - 1) {
        if (getBlock(vec3i(bx, y, bz)) != 0u) {
            top = y + 1;
            break;
        }
    }
    return f32(top);
}
```

从世界顶部往下扫，找到第一个实心方块，返回它的**顶面高度**（`y + 1`，因为是方块的上边界）。

这比"高度图"更通用：玩家挖了一个坑，`groundAt` 会返回坑底的高度，人自然就掉下去了，**不需要写重力、不需要写碰撞、不需要速度积分**。

代价是每次移动要做一次 `O(WY)` 的扫描，也就是最多 16 次 `getBlock`。相对 DDA 的开销可以忽略，而且它每帧只在 `(0,0)` 线程里跑几次。

### 6.2 移动与台阶限制

```wgsl
var input = vec3f(0.0);
if (keyDown(87)) { input = input + fw * vec3f(1.0, 0.0, 1.0); }  // W：沿视线水平前进
if (keyDown(83)) { input = input - fw * vec3f(1.0, 0.0, 1.0); }  // S
if (keyDown(68)) { input = input + rgt; }                        // D
if (keyDown(65)) { input = input - rgt; }                        // A

if (length(input) > 1e-5) {
    input = normalize(input) * SPEED * time.delta;
    let nx = pos.x + input.x;
    let nz = pos.z + input.z;
    let g  = groundAt(nx, nz);
    if (g - (pos.y - EYE) <= 1.2) {        // ← 台阶限制
        pos = vec3f(nx, g + EYE, nz);
    }
} else {
    pos = vec3f(pos.x, groundAt(pos.x, pos.z) + EYE, pos.z);
}
```

几个细节：

**`fw * vec3f(1.0, 0.0, 1.0)`**：把前向量在水平面投影。抬着头按 W 不会飞起来，低头按 W 也不会钻地。

**`normalize(input) * SPEED * time.delta`**：先归一化再乘速度，保证斜着走（W+D）不会比直走快（否则对角线速度是 `√2` 倍）。乘 `time.delta` 让速度与帧率无关。

**`g - (pos.y - EYE) <= 1.2`**：`pos.y - EYE` 是脚底高度，`g` 是目标位置的地面高度，差值就是"需要抬多少"。小于 1.2 格时允许走上去（自动上台阶），超过就当墙挡住。

这个判据同时处理了"上坡"和"撞墙"：小坡能上，大石头挡住。

**静止时也吸附地面**：`else` 分支。否则玩家挖掉自己脚下的方块后会悬空（因为位置只在移动时更新）。

### 6.3 眼睛高度

```wgsl
const EYE : f32 = 1.62;
```

`pos` 存的是**眼睛位置**，脚底是 `pos.y - EYE`。1.62 是成年人的眼睛高度（米），换成 1 格 = 1 米后刚好比一格高一点，视觉效果接近 Minecraft。

---

## 7. 单 pass 内的竞态与 `time.frame` 防重入

这是这个实现里最微妙的部分。

### 7.1 问题

物理必须**每帧只推进一次**。但：

- 一次 dispatch 里，每个像素都是一个独立线程
- 线程之间**没有执行顺序保证**
- 如果所有线程都读位置、都加位移、都写回，位置会被推进 `width × height` 次

所以只能让一个线程写。但**读取的线程也需要知道新位置**用于渲染——否则画面会用旧位置，出现一帧延迟。

### 7.2 解决：帧号标记

```wgsl
// 读：帧号等于当前帧 -> 说明本帧已经推进过，直接用存下来的位置
if (world[STATE + 3u] != time.frame) {
    ... 计算新位置 ...
    if (id.x == 0u && id.y == 0u) { storePos(pos, time.frame); }
}
```

关键在于**所有线程算出的结果必然相同**：输入一样（同一份位置、同一份按键、同一个 `time.delta`），运算也是纯函数。

于是无论哪个线程先跑：

| 情况 | 该线程看到的帧号 | 行为 | 结果 |
|---|---|---|---|
| 先跑的线程 | 旧帧号 | 自己算一遍新位置 | 正确 |
| 后跑的线程 | 新帧号（(0,0) 已写） | 直接读存下来的位置 | 正确 |

两条路径得到同一个 `pos`，所以**画面不会撕裂**，也不会重复推进。

### 7.3 残余风险

`storePos` 分四次写：`pos.x`、`pos.y`、`pos.z`、帧号。这四次不是原子操作。

如果某个线程恰好在 (0,0) 写完 `pos.x` 但还没写完 `pos.z` 时读到，它会拿到"半个新位置"，导致该像素一帧的位置错误——表现为画面上一小块区域闪一下。

概率很低（窗口是纳秒级），而且只影响一帧，所以没有处理。真要消除，可以把位置打包成 `vec4f` 写进一个 `vec4<f32>` 的 storage 别名，或者接受两帧延迟（写下一帧的槽位）。

---

## 8. 挖掘与放置

```wgsl
if (id.x == 0u && id.y == 0u && mouse.click != 0) {
    let mh = raycast(pos, fw);          // 从眼睛沿视线中心打一条
    if (mh.hit) {
        if (mouse.click == 1) {
            setBlock(mh.cell, 0u);                  // 左键：把命中的格子变空气
        } else {
            var mat = 1u;
            if (keyDown(50)) { mat = 2u; }          // 按住 2 放石头，否则放土
            setBlock(mh.cell + mh.nrm, mat);        // 右键：放在命中面的外侧
        }
    }
}
```

**为什么是 `cell + nrm`**：`nrm` 是命中面的法线，指向格子外面。所以 `cell + nrm` 正好是"贴着被点中的那个面、在外侧"的格子。这正是 Minecraft 放置方块的位置。

**为什么只有 `(0,0)` 线程做**：编辑是"单点操作"，多线程做会重复写（虽然写得一样，但会造成 `getBlock` 读到的状态在同一帧内不一致，导致画面部分区域显示"挖之前"、部分显示"挖之后"）。用单个线程保证语义清晰。

**为什么用 `fw` 而不是某条像素射线**：准星固定在屏幕中心，`fw` 就是屏幕中心的射线方向，和 `(0,0)` 线程负责的像素无关。

---

## 9. 性能分析

### 9.1 开销在哪

每次渲染的代价大致是：

```
像素数 × (AABB 测试 + DDA 平均步数 × (getBlock 开销))
```

`getBlock` 的开销包括：一次分支、一次 storage 读、以及**未命中时的 3 组 sin/cos**。

`heightAt` 里的三角函数是这个实现最大的性能黑洞——它在 DDA 内层循环里被反复调用，而绝大多数读到的都是"未改动"的格子。

### 9.2 数量级估算

1080p ≈ 200 万像素，平均每条射线穿过 20~50 个格子，最坏 256：

- 典型：约 **5000 万 ~ 1 亿次** storage 读 + 同等数量的三角函数
- 这个量级在桌面独显上能跑，在集显或移动端会很吃力

### 9.3 优化方向（按性价比排序）

**① 缓存 `heightAt`**
如果地形不变，`heightAt` 对同一 `(x,z)` 的结果是固定的。可以在每帧第一次 `getBlock` 时把整列高度预计算进 `var<workgroup>`，但这需要世界划分到 workgroup，改造量较大。

**② 预计算高度表写进 storage**
把 64×64 的高度表（4096 个 u32，16 KB）在初始化时算好存进 buffer，`getBlock` 直接查表。**这是收益最大、改动最小的一步**：把 DDA 里的三角函数全部换成一次内存读。

**③ 降低 DDA 步数上限**
`256` 是保守值。如果相机视野内很少看到 100 格以外的东西，改成 128 能减少最坏情况的浪费（但仍需覆盖对角）。

**④ 世界分区 / 八叉树**
在 DDA 之前先用一个粗粒度的占用表（比如 8×8×8 的块级表）快速跳过空气。改动大，收益取决于场景的空洞程度。

**⑤ 把 `world` 换成 3D 纹理**
`textureLoad` 的三线性过滤和硬件缓存可能更好，但 `3D` 纹理在 WebGPU 里有尺寸和格式限制，且会失去 `bitcast` 存状态的能力。

---

## 10. 已知限制

| 限制 | 原因 | 影响 |
|---|---|---|
| 视角由鼠标绝对位置决定 | 没有指针锁定，用位置映射代替 | 瞄准和转视角是同一个动作，与挖掘冲突 |
| 鼠标到屏幕边缘后视角卡住 | 同上 | 无法无限转身 |
| 没有跳跃 | 用 `groundAt` 吸附代替重力 | 上不去高于 1.2 格的台阶 |
| 没有真正的碰撞盒 | 只有垂直吸附 + 水平单点检测 | 贴墙走会穿过墙角（只检查目的地中心点） |
| 世界边缘会掉出去 | `groundAt` 在界外返回 0 | 走到世界外会掉到 `y = EYE` |
| buffer 大小要手动配 | `#storage` 不声明长度 | 配小了编辑写不进去（静默丢弃） |
| 单 pass 依赖帧号 | 没有 pass 间同步 | 极端情况下一帧的画面撕裂 |
| 没有阴影/AO | 只有面朝向明暗 | 方块交界处缺少接触阴影 |

---

## 11. 如果想继续扩展

按实现难度排序：

1. **相对视角**：把 `yaw/pitch` 存进状态槽，每帧累加 `mouse.delta`。约 20 行改动，解决最大的操作问题。
2. **高度表预计算**：性能立刻提升一个量级。
3. **跳跃**：增加 `velY`，`groundAt` 只做碰撞而不是硬吸附。
4. **方块类型选择**：用 1/2/3 键选土/石/原木，扩展 `mat`。
5. **准星高亮**：把命中格子的描边画出来，操作反馈会好很多。
6. **太阳方向光**：`dot(nrm, lightDir)` 代替常数 `shade`，几乎零成本。
7. **雾**：按 `h.t` 混入天空色，遮蔽远处 DDA 截断的边界。

---

## 附：一次完整帧的执行顺序

```
dispatch 开始
│
├─ 每个线程：
│   ├─ 读 world[STATE..STATE+3]  → pos, frameTag
│   ├─ 若 pos 未初始化 → 用出生点
│   ├─ 从 mouse.pos 算 yaw/pitch → 得到 fw/rt/up
│   │
│   ├─ if frameTag != time.frame:
│   │     ├─ 读 WASD → 水平移动向量
│   │     ├─ groundAt(目标点) → 吸附 + 台阶判定
│   │     └─ if id == (0,0): storePos(pos, time.frame)
│   │
│   ├─ if id == (0,0) && click: 沿 fw 打射线 → setBlock
│   │
│   ├─ 沿 rd 打射线 → 命中体素
│   │     ├─ DDA: AABB 裁剪 → 逐格步进 → 命中返回 cell/nrm/t
│   │     ├─ 按 nrm 选 uv → 采样 channel0/1
│   │     └─ 按 nrm 选亮度
│   │
│   ├─ 未命中 → 天空渐变
│   ├─ 屏幕中心 2px → 白色准星
│   └─ textureStore(screen, id.xy, col)
│
dispatch 结束
```
