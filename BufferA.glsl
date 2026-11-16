// ============================================================
//  Buffer A —— 世界状态 + 玩家物理 + 挖掘/放置
//  iChannel0 = Buffer A（自反馈）
//  iChannel3 = Keyboard（可选，没有也不影响出画面）
// ============================================================

#define WX 64
#define WY 16
#define WZ 64
#define NBLK 65536
#define EYE   1.62
#define SPEED 6.0

bool keyDown(int c) {
    return texture(iChannel3, vec2((float(c) + 0.5) / 256.0, 0.5 / 3.0)).x > 0.5;
}

int heightAt(int x, int z) {
    float fx = float(x), fz = float(z);
    float hf = 5.0 + 2.5 * sin(fx * 0.18) * cos(fz * 0.15)
                   + 1.8 * sin(fx * 0.07 + fz * 0.11)
                   + 0.6 * sin(fx * 0.53) * sin(fz * 0.47);
    return int(clamp(hf, 0.0, float(WY - 1)));
}

int idxOf(ivec3 q) { return q.x + q.z * WX + q.y * WX * WZ; }

ivec2 blkTexel(int b) {
    int w = int(iResolution.x);
    int n = b + 3;
    return ivec2(n % w, n / w);
}

int blockAt(ivec3 q) {
    if (q.x < 0 || q.y < 0 || q.z < 0 || q.x >= WX || q.y >= WY || q.z >= WZ) return 0;
    float e = texelFetch(iChannel0, blkTexel(idxOf(q)), 0).r;
    int v = int(e * 4.0 + 0.5) - 1;
    if (v >= 0) return v;
    int h = heightAt(q.x, q.z);
    if (q.y < h) return 2;
    if (q.y == h) return 1;
    return 0;
}

float groundAt(float x, float z) {
    int bx = int(floor(x)), bz = int(floor(z));
    int top = 0;
    for (int y = WY - 1; y >= 0; y--) {
        if (blockAt(ivec3(bx, y, bz)) != 0) { top = y + 1; break; }
    }
    return float(top);
}

struct Hit { bool hit; ivec3 cell; ivec3 nrm; float t; };

Hit raycast(vec3 ro, vec3 rd) {
    Hit h;
    h.hit = false; h.cell = ivec3(0); h.nrm = ivec3(0); h.t = 0.0;

    vec3 inv = 1.0 / rd;
    vec3 ta = (vec3(0.0) - ro) * inv;
    vec3 tb = (vec3(float(WX), float(WY), float(WZ)) - ro) * inv;
    vec3 lo = min(ta, tb), hi = max(ta, tb);
    float tn = max(max(lo.x, lo.y), lo.z);
    float tf = min(min(hi.x, hi.y), hi.z);
    if (tf < max(tn, 0.0)) return h;

    float t = max(tn, 0.0) + 1e-4;
    vec3 p0 = ro + rd * t;
    ivec3 c = ivec3(floor(p0));

    ivec3 stp = ivec3(sign(rd));
    vec3 dd = abs(1.0 / rd);
    vec3 tt;
    tt.x = abs(rd.x) > 1e-8 ? t + (float(c.x) + step(0.0, rd.x) - p0.x) / rd.x : 1e30;
    tt.y = abs(rd.y) > 1e-8 ? t + (float(c.y) + step(0.0, rd.y) - p0.y) / rd.y : 1e30;
    tt.z = abs(rd.z) > 1e-8 ? t + (float(c.z) + step(0.0, rd.z) - p0.z) / rd.z : 1e30;

    ivec3 n = ivec3(0);
    for (int k = 0; k < 256; k++) {
        if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= WX || c.y >= WY || c.z >= WZ) break;
        if (blockAt(c) != 0) { h.hit = true; h.cell = c; h.nrm = n; h.t = t; return h; }
        if (tt.x < tt.y && tt.x < tt.z) { t = tt.x; c.x += stp.x; tt.x += dd.x; n = ivec3(-stp.x, 0, 0); }
        else if (tt.y < tt.z)           { t = tt.y; c.y += stp.y; tt.y += dd.y; n = ivec3(0, -stp.y, 0); }
        else                            { t = tt.z; c.z += stp.z; tt.z += dd.z; n = ivec3(0, 0, -stp.z); }
    }
    return h;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    int w = int(iResolution.x);
    int n = int(fragCoord.x) + int(fragCoord.y) * w;

    // ================= 状态 texel =================
    if (n < 2) {
        vec4 s0 = texelFetch(iChannel0, ivec2(0, 0), 0);
        vec4 s1 = texelFetch(iChannel0, ivec2(1, 0), 0);

        vec3  pos      = vec3(s0.r * 64.0, s0.g * 16.0, s0.b * 64.0);
        float prevDown = s0.a;
        if (abs(pos.x) + abs(pos.y) + abs(pos.z) < 1e-3) pos = vec3(32.0, 14.0, 6.0);

        vec2 res = iResolution.xy;
        vec2 m   = iMouse.xy / res;
        if (iMouse.xy.x <= 0.0 && iMouse.xy.y <= 0.0) m = vec2(0.5);

        float yaw   = (m.x - 0.5) * 4.0;
        float pitch = (m.y - 0.5) * 1.4;
        vec3 fw  = vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch));
        vec3 rgt = vec3(-cos(yaw), 0.0, sin(yaw));

        // ---- 移动：键盘 + 鼠标边缘双通道 ----
        vec3 mv = vec3(0.0);
        if (keyDown(87)) mv += fw * vec3(1.0, 0.0, 1.0);
        if (keyDown(83)) mv -= fw * vec3(1.0, 0.0, 1.0);
        if (keyDown(68)) mv += rgt;
        if (keyDown(65)) mv -= rgt;

        if (m.y > 0.92) mv += fw * vec3(1.0, 0.0, 1.0);
        if (m.y < 0.08) mv -= fw * vec3(1.0, 0.0, 1.0);
        if (m.x > 0.94) mv += rgt;
        if (m.x < 0.06) mv -= rgt;

        if (length(mv) > 1e-5) {
            mv = normalize(mv) * SPEED * max(iTimeDelta, 0.0);
            vec3 np = pos + mv;
            float g = groundAt(np.x, np.z);
            if (g - (pos.y - EYE) <= 1.2) pos = vec3(np.x, g + EYE, np.z);
        } else {
            pos = vec3(pos.x, groundAt(pos.x, pos.z) + EYE, pos.z);
        }

        // ---- 点击：单击挖 / 双击放 ----
        bool down  = iMouse.z > 0.0;
        bool click = down && prevDown < 0.5;

        int lastClick = int(s1.a * 4096.0 + 0.5);
        int curFrame  = iFrame % 4096;
        bool dbl = (s1.a > 0.0) && (curFrame - lastClick) > 0 && (curFrame - lastClick) < 12;

        int tgt = -1, action = 0, mat = 1;
        if (click) {
            Hit hh = raycast(pos, fw);
            if (hh.hit) {
                bool place = dbl || keyDown(16);
                if (place) { action = 2; mat = keyDown(50) ? 2 : 1; tgt = idxOf(hh.cell + hh.nrm); }
                else       { action = 1; tgt = idxOf(hh.cell); }
            }
            lastClick = curFrame;
        }

        if (n == 0) fragColor = vec4(pos.x / 64.0, pos.y / 16.0, pos.z / 64.0, down ? 1.0 : 0.0);
        else        fragColor = vec4(float(tgt + 1) / 65536.0, float(action) / 4.0,
                                     float(mat) / 4.0, float(lastClick) / 4096.0);
        return;
    }

    // ================= 方块 texel =================
    int b = n - 3;
    if (b >= NBLK) { fragColor = vec4(0.0); return; }

    float prev = texelFetch(iChannel0, ivec2(fragCoord), 0).r;

    vec4 s1    = texelFetch(iChannel0, ivec2(1, 0), 0);
    int  tgt    = int(s1.r * 65536.0 + 0.5) - 1;
    int  action = int(s1.g * 4.0 + 0.5);
    int  mat    = int(s1.b * 4.0 + 0.5);

    int outV = -1;
    if (action != 0 && b == tgt) outV = (action == 1) ? 0 : mat;

    fragColor = (outV < 0)
              ? vec4(prev, 0.0, 0.0, 1.0)
              : vec4(float(outV + 1) / 4.0, 0.0, 0.0, 1.0);
}
