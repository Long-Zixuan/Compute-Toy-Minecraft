// ============================================================
//  Image —— 体素射线步进 + 着色 + 准星
//  iChannel0 = Buffer A
// ============================================================

#define WX 64
#define WY 16
#define WZ 64
#define FOV_K 1.0
#define EYE   1.62

int heightAt(int x, int z) {
    float fx = float(x), fz = float(z);
    float hf = 5.0 + 2.5 * sin(fx * 0.18) * cos(fz * 0.15)
                   + 1.8 * sin(fx * 0.07 + fz * 0.11)
                   + 0.6 * sin(fx * 0.53) * sin(fz * 0.47);
    return int(clamp(hf, 0.0, float(WY - 1)));
}

int idxOf(ivec3 q) { return q.x + q.z * WX + q.y * WX * WZ; }

ivec2 blkTexel(int b) {
    int w = textureSize(iChannel0, 0).x;
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
    vec2 res = iResolution.xy;

    vec4 s0 = texelFetch(iChannel0, ivec2(0, 0), 0);
    vec4 s1 = texelFetch(iChannel0, ivec2(1, 0), 0);

    bool stateOK = length(s0.rgb) > 1e-4;

    vec3 pos = vec3(s0.r * 64.0, s0.g * 16.0, s0.b * 64.0);
    if (!stateOK) pos = vec3(32.0, 14.0, 6.0);
    pos.y = groundAt(pos.x, pos.z) + EYE;

    int tgt = int(s1.r * 65536.0 + 0.5) - 1;

    vec2 m = iMouse.xy / res;
    if (iMouse.xy.x <= 0.0 && iMouse.xy.y <= 0.0) m = vec2(0.5);

    float yaw   = (m.x - 0.5) * 4.0;
    float pitch = (m.y - 0.5) * 1.4;
    vec3 fw = vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch));
    vec3 rt = normalize(cross(fw, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(rt, fw);

    vec2 uv = (fragCoord / res - 0.5) * vec2(res.x / res.y, 1.0);
    vec3 rd = normalize(uv.x * rt + uv.y * up + FOV_K * fw);

    Hit h = raycast(pos, rd);
    vec3 col;

    if (h.hit) {
        vec3 p = pos + rd * h.t;
        vec3 f = p - vec3(h.cell);

        vec2 tuv;
        if (h.nrm.x != 0)      tuv = vec2(f.z, 1.0 - f.y);
        else if (h.nrm.y != 0) tuv = vec2(f.x, f.z);
        else                   tuv = vec2(f.x, 1.0 - f.y);

        int id = blockAt(h.cell);

        // 顶面用草色，其余面用土/石
        vec3 base = (id == 1 && h.nrm.y > 0)
                  ? vec3(0.416, 0.659, 0.310)
                  : (id == 2 ? vec3(0.500, 0.500, 0.500) : vec3(0.545, 0.353, 0.169));

        // 格内 8x8 伪噪声 + 每方块色调偏移，模拟像素画贴图
        vec2 q = floor(tuv * 8.0);
        float n1 = fract(sin(dot(q, vec2(12.9898, 78.233))) * 43758.5453);
        float n2 = fract(sin(dot(vec2(h.cell.xz), vec2(39.3468, 11.135))) * 24634.6345);
        col = base * (0.82 + 0.30 * n1) * (0.92 + 0.16 * n2);

        // 面朝向的廉价明暗
        float shade = 0.80;
        if (h.nrm.y != 0)      shade = (h.nrm.y > 0) ? 1.0 : 0.5;
        else if (h.nrm.x != 0) shade = 0.70;
        col *= shade;

        // 瞄准中的方块高亮
        if (idxOf(h.cell) == tgt) col = mix(col, vec3(1.0), 0.25);
    } else {
        col = mix(vec3(0.55, 0.72, 0.95), vec3(0.15, 0.38, 0.80),
                  clamp(rd.y * 1.5 + 0.5, 0.0, 1.0));
    }

    // 屏幕中心准星
    vec2 dc = abs(fragCoord - res * 0.5);
    if (max(dc.x, dc.y) < 2.0) col = vec3(1.0);

    fragColor = vec4(col, 1.0);
}
