"""
Faz 1 - Fizik çekirdeği (numpy, popülasyon vektörize).

Model
-----
* Canlı bir AĞAÇTIR: nokta 0 = kafa (kök). Her i>=1 noktası için bir "kemik" vardır:
  parent[i] -> i. parent[i] < i olmak zorundadır (ağaç sırası).
* Kütle noktalarda (eklemlerde) bulunur, kemikler kütlesizdir.
* Kemik uzunlukları kısıtlıdır (PBD / Gauss-Seidel).
* Kemik 1 = "boyun/omurga": kafanın yönünü tanımlar (gövde açısı phi = açı(kemik 1) - theta0[1]).
  Bağımsız bir kafa-açısı serbestliği YOKTUR; gövde açısı tamamen geometriden ve fizikten doğar.
  Kemik 1'in kendi motoru yoktur (pasif); en fazla 23 motorlu eklem + 1 pasif omurga.
* Eklem i (i>=2 veya ...) = kemik i ile referans kemiği arasındaki açı.
    - parent[i] == 0 ise referans = kemik 1
    - aksi halde referans = parent[i] noktasının kendi kemiği.
  Eklem torku iki taraflı kuvvet çifti olarak uygulanır (net kuvvet sıfır, açısal momentum korunur).
* Su: kemik başına anizotropik (normal >> teğet) karesel sürükleme, 2 alt-segmentle.
* Dış hile yok: canlıya yalnızca eklem torkları ve su kuvvetleri etki eder.

Tüm diziler (B, N) veya (B, N, 2) şeklindedir; B = popülasyon, N = MAX_POINTS (25 = kafa + 24 uzuv).
Kullanılmayan noktalar `active=False` ile maskelenir.

Not: Eklem açısı theta_rel, dinlenme açısına (theta0) göre ölçülür ve [jlo, jhi] ile sınırlanır
(jlo < 0 < jhi). Böylece ±pi sarma sorunu çıkmaz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

MAX_LIMBS = 24
MAX_POINTS = MAX_LIMBS + 2     # kafa + 24 uzuv + 1 'burun' noktası (kafa yönü)
NOSE_L, NOSE_M, NOSE_W = 0.5, 1.0, 0.1   # burun: kafaya bağlı sabit uzunluklu, motorsuz kemik; yönü = gövde yönü
# Fizik sürümü: sayısal sonuçları değiştiren her düzeltmede artırılır.
#   1: dt=0.01 (hafif uzuv + güçlü motor kombinasyonlarında sayısal enerji sömürüsüne açıktı)
#   2: dt=0.005 + tork sınırı
#   3: DOĞRUSAL (viskoz) sürükleme, anizotropik (düşük Reynolds sayısı rejimi): dalgalanarak yüzme çok daha
#      verimli (ideal dalga 0.12 -> ~1 m/s) ve motorlar kapanınca canlı ~2 sn'de duruyor (eski: 14 sn).
#   4: eklem sınırı yayı/sönümü ve TOPLAM eklem torku ataletle sınırlı (indirgenmiş atalet): hafif+kısa kemik zincirlerinde
#      sınır yayı sayısal olarak patlıyordu (tork on binlere, eklem hızı yüzlerce rad/s, sonuç dt'ye bağlı). Gövde yönü
#      (burun) modu da bu sürümle geldi (eski mod davranışı aynı kalır).
PHYSICS_VERSION = 4


# --------------------------------------------------------------------------- #
#  Parametreler ve veri yapıları
# --------------------------------------------------------------------------- #
@dataclass
class PhysicsParams:
    dt: float = 0.005
    rho: float = 1.0          # su yoğunluğu ölçeği
    cn: float = 0.0           # KARESEL sürükleme: normal yön (sürüm 3'te kapalı; yüksek Reynolds rejimi için)
    ct: float = 0.0           # karesel sürükleme: teğet yön
    kn_lin: float = 60.0      # DOĞRUSAL (viskoz) sürükleme: kemiğe DİK yön katsayısı
    kt_lin: float = 3.0       # kemiğe PARALEL yön (kn_lin'in 1/20'si -> güçlü anizotropi = itki)
    c_point: float = 0.05     # nokta (gövde) izotropik karesel sürüklemesi (küçük: gövde akıntıya karşı ince)
    alpha_max: float = 250.0  # azami açısal ivme (rad/s^2): tork = min(komut*tau_max, alpha_max*atalet)
                              # hafif uzva aşırı güçlü motor = zaman adımının çözemediği sayısal rejim
    k_limit: float = 40.0     # eklem sınırı yay sertliği (tork_max ile çarpılır)
    c_limit: float = 4.0      # eklem sınırı sönümü
    iterations: int = 4       # PBD tur sayısı (ileri+geri süpürme)
    safe_joint: bool = True   # sürüm 4: eklem torku/sınır yayı atalete göre sınırlı (False -> sürüm 3 davranışı)
    w_limit: float = 80.0     # sınır yayının izin verilen en büyük doğal frekansı (rad/s); dt=0.005'te w*dt=0.4


@dataclass
class Bodies:
    """Bir popülasyonun (B canlı) sabit anatomisi. Faz 2'de BodyGenome'dan üretilecek."""
    parent: np.ndarray        # (B,N) int
    length: np.ndarray        # (B,N) kemik uzunluğu (i>=1)
    width: np.ndarray         # (B,N) kemik kalınlığı (sürükleme yüzeyi)
    mass: np.ndarray          # (B,N) nokta kütlesi
    theta0: np.ndarray        # (B,N) dinlenme açısı (referans kemiğe göre)
    jlo: np.ndarray           # (B,N) alt sınır (<=0) dinlenmeye göre
    jhi: np.ndarray           # (B,N) üst sınır (>=0)
    tau_max: np.ndarray       # (B,N) tork kapasitesi
    jdamp: np.ndarray         # (B,N) eklem sönümü
    active: np.ndarray        # (B,N) bool; [:,0] her zaman True
    sub_mass: np.ndarray = None   # (B,N) nokta + alt ağaç kütlesi (atalet tahmini)
    legacy_spine: np.ndarray = None  # (B,) bool: True = ESKİ mod (kemik 1 pasif omurga, gövde yönünü o belirler);
                                     # False = gövde yönü = kafa-burun kemiği (son nokta N-1), tüm uzuvlar motorlu olabilir

    @property
    def B(self):
        return self.parent.shape[0]

    @property
    def N(self):
        return self.parent.shape[1]

    @property
    def bone_active(self):
        ba = self.active.copy()
        ba[:, 0] = False
        return ba


@dataclass
class State:
    x: np.ndarray       # (B,N,2)
    v: np.ndarray       # (B,N,2)
    phi: np.ndarray     # (B,) gövde açısı (türetilmiş)
    omega: np.ndarray   # (B,) gövde açısal hızı (türetilmiş)
    t: float = 0.0
    theta: np.ndarray = None   # (B,N) son eklem açıları (dinlenmeye göre)
    jvel: np.ndarray = None    # (B,N) son eklem açısal hızları
    tau: np.ndarray = None     # (B,N) son uygulanan toplam eklem torku


# --------------------------------------------------------------------------- #
#  Canlı kurma yardımcıları
# --------------------------------------------------------------------------- #
def make_bodies(specs: list[dict], n_points: int = MAX_POINTS) -> Bodies:
    """
    specs: her canlı için dict:
        parent: list[int]            uzunluğu n (nokta 0 dahil; parent[0] göz ardı edilir)
        length, width, mass, theta0, jlo, jhi, tau_max, jdamp: uzunluğu n listeler (indeks 0 göz ardı)
    """
    B, N = len(specs), n_points
    leg = np.array([bool(s.get("legacy_spine", True)) for s in specs], dtype=bool)
    z = lambda: np.zeros((B, N))
    parent = np.zeros((B, N), dtype=np.int64)
    arr = {k: z() for k in ("length", "width", "mass", "theta0", "jlo", "jhi", "tau_max", "jdamp")}
    active = np.zeros((B, N), dtype=bool)
    for b, s in enumerate(specs):
        n = len(s["parent"])
        assert n <= (N if leg[b] else N - 1), "en fazla %d uzuv" % (N - 1 - (0 if leg[b] else 1))
        for i in range(1, n):
            assert 0 <= s["parent"][i] < i, "ağaç sırası: parent[i] < i olmalı"
        parent[b, :n] = s["parent"]
        for k in arr:
            arr[k][b, :n] = s[k]
        active[b, :n] = True
        if not leg[b]:                      # burun: kafa yönünü taşıyan son nokta (kafaya bağlı)
            k = N - 1
            parent[b, k] = 0
            arr["length"][b, k], arr["width"][b, k], arr["mass"][b, k] = NOSE_L, NOSE_W, NOSE_M
            active[b, k] = True
    # kullanılmayan noktalar: güvenli değerler
    arr["length"][~active] = 1.0
    arr["mass"][~active] = 1.0
    arr["width"][~active] = 0.0
    sub = arr["mass"].copy()
    for b_ in range(B):
        for i in range(N - 1, 0, -1):
            if active[b_, i]:
                sub[b_, parent[b_, i]] += sub[b_, i]
    return Bodies(parent=parent, active=active, sub_mass=sub, legacy_spine=leg, **arr)


def init_state(bodies: Bodies, pos=None, heading=None) -> State:
    """Dinlenme geometrisinden başlangıç durumu. pos (B,2), heading (B,) opsiyonel."""
    B, N = bodies.B, bodies.N
    pos = np.zeros((B, 2)) if pos is None else np.asarray(pos, float).reshape(B, 2)
    heading = np.zeros(B) if heading is None else np.asarray(heading, float).reshape(B)
    x = np.zeros((B, N, 2))
    x[:, 0] = pos
    alpha = np.zeros((B, N))
    ar = np.arange(B)
    leg = _leg(bodies)
    for i in range(1, N):
        p = bodies.parent[:, i]
        if i == 1:
            ref = heading
        else:
            ref = np.where(p == 0, np.where(leg, alpha[:, 1], heading), alpha[ar, p])
        alpha[:, i] = ref + bodies.theta0[:, i]
        d = bodies.length[:, i, None] * np.stack([np.cos(alpha[:, i]), np.sin(alpha[:, i])], -1)
        x[:, i] = x[ar, p] + d
    # kullanılmayan noktalar kafada dursun
    x[~bodies.active] = np.broadcast_to(pos[:, None, :], x.shape)[~bodies.active]
    return State(x=x, v=np.zeros_like(x), phi=heading.copy(), omega=np.zeros(B), t=0.0,
                 theta=np.zeros((B, N)), jvel=np.zeros((B, N)), tau=np.zeros((B, N)))


# --------------------------------------------------------------------------- #
#  Çekirdek fonksiyonlar
# --------------------------------------------------------------------------- #
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _leg(bd):
    return np.ones(bd.B, dtype=bool) if bd.legacy_spine is None else bd.legacy_spine


def _heading(x, v, bd):
    """Gövde açısı ve açısal hızı. Eski mod: kemik 1 yönü (-theta0). Yeni mod: kafadan burun kemiğine."""
    leg = _leg(bd)
    k = np.where(leg, 1, bd.N - 1)
    ar = np.arange(bd.B)
    d = x[ar, k] - x[:, 0]
    dv = v[ar, k] - v[:, 0]
    off = np.where(leg, bd.theta0[:, 1], 0.0)
    phi = np.arctan2(d[:, 1], d[:, 0]) - off
    om = (d[:, 0] * dv[:, 1] - d[:, 1] * dv[:, 0]) / np.maximum((d ** 2).sum(-1), 1e-9)
    return phi, om


def _scatter(idx, vals, B, N):
    """(B,N) hedef indekslerine (B,N,2) değerleri toplayarak yaz -> (B,N,2)."""
    flat = (np.arange(B)[:, None] * N + idx).ravel()
    out = np.empty((B * N, 2))
    for c in range(2):
        out[:, c] = np.bincount(flat, weights=vals[..., c].ravel(), minlength=B * N)
    return out.reshape(B, N, 2)


def _gather(arr, idx):
    """arr (B,N,...) içinden idx (B,N) ile satır seç."""
    if arr.ndim == 3:
        return np.take_along_axis(arr, idx[..., None], axis=1)
    return np.take_along_axis(arr, idx, axis=1)


SPINE_BONE = 1   # gövde yönünü tanımlayan kemik


def motor_mask(bd: Bodies) -> np.ndarray:
    """(B,N) bool: hangi kemiklerin motorlu eklemi var. Beyin G/Ç ve genom BURADAN okur.
    Pasif omurgayı değiştirmek için bu fonksiyon ve joint_kinematics/step içindeki
    referans-kemik tanımı (refidx) yeterlidir."""
    m = bd.bone_active.copy()
    leg = _leg(bd)
    m[:, SPINE_BONE] = ~leg & m[:, SPINE_BONE]         # eski mod: kemik 1 pasif omurga; yeni modda motorlu
    m[:, bd.N - 1] = leg & m[:, bd.N - 1]              # yeni mod: son nokta = burun (motorsuz)
    return m


def joint_kinematics(st: State, bd: Bodies):
    """Eklem açısı (dinlenmeye göre) ve açısal hızı, kemik yönleri vb."""
    par = bd.parent
    xp = _gather(st.x, par)
    vp = _gather(st.v, par)
    d = st.x - xp
    L = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
    u = d / L[..., None]
    n = np.stack([-u[..., 1], u[..., 0]], -1)
    alpha = np.arctan2(d[..., 1], d[..., 0])
    dv = st.v - vp
    wb = (d[..., 0] * dv[..., 1] - d[..., 1] * dv[..., 0]) / L ** 2
    # referans kemik: kafaya bağlı uzuvlar eski modda kemik 1'e, yeni modda burun kemiğine (N-1) göre eklemlenir
    refidx = np.where(par == 0, np.where(_leg(bd)[:, None], 1, bd.N - 1), par)
    ref = _gather(alpha, refidx)
    wref = _gather(wb, refidx)
    theta = _wrap(alpha - ref - bd.theta0)
    jvel = wb - wref
    has_joint = motor_mask(bd)
    theta = np.where(has_joint, theta, 0.0)
    jvel = np.where(has_joint, jvel, 0.0)
    return dict(d=d, L=L, u=u, n=n, alpha=alpha, wb=wb, theta=theta, jvel=jvel,
                has_joint=has_joint, refidx=refidx)


def step(st: State, bd: Bodies, ctrl: np.ndarray, pp: PhysicsParams, drag: bool = True):
    """
    Bir fizik adımı. ctrl: (B,N) in [-1,1]; ctrl[:,i] = eklem i'in (kemik i) motor komutu.
    Dönüş: yok (st yerinde güncellenir).
    """
    B, N = bd.B, bd.N
    dt = pp.dt
    ar = np.arange(B)
    ba = bd.bone_active
    par = bd.parent
    kin = joint_kinematics(st, bd)
    L, u, n, theta, jvel = kin["L"], kin["u"], kin["n"], kin["theta"], kin["jvel"]
    hj, refidx = kin["has_joint"], kin["refidx"]

    # ---- eklem torku ----
    tau = np.clip(ctrl, -1.0, 1.0) * bd.tau_max
    tau_cap = pp.alpha_max * np.maximum(bd.sub_mass * L ** 2, 1e-4)
    tau = np.clip(tau, -tau_cap, tau_cap)
    tau = tau - bd.jdamp * jvel
    over = np.maximum(theta - bd.jhi, 0.0)
    under = np.maximum(bd.jlo - theta, 0.0)
    viol = (over > 0) | (under > 0)
    if pp.safe_joint:
        # İndirgenmiş atalet: kemiğin ve referans kemiğin uç-nokta ataletlerinin seri birleşimi (alt sınır tahmini).
        L_ref = _gather(L, refidx)
        I_i = bd.mass * L ** 2
        I_r = _gather(bd.mass, refidx) * L_ref ** 2
        if not _leg(bd).all():                      # burun referanslı eklemler: kafa+burun çiftinin indirgenmiş ataleti
            mh, mn = bd.mass[:, 0], bd.mass[:, N - 1]
            i_n = (mh * mn / (mh + mn) * bd.length[:, N - 1] ** 2)[:, None]
            I_r = np.where((refidx == N - 1) & ~_leg(bd)[:, None], i_n, I_r)
        I_red = np.maximum(I_i * I_r / np.maximum(I_i + I_r, 1e-9), 1e-4)
        k_lim = np.minimum(pp.k_limit * bd.tau_max, pp.w_limit ** 2 * I_red)
        c_lim = np.minimum(np.maximum(pp.c_limit * bd.tau_max * 0.1, 1.4 * np.sqrt(k_lim * I_red)), 0.9 * I_red / dt)
        tau += k_lim * (under - over)
        tau -= np.where(viol, c_lim * jvel, 0.0)
        cap2 = pp.alpha_max * I_red
        tau = np.clip(tau, -cap2, cap2)             # motor + sönüm + sınır: toplam tork açısal ivmeyi sınırlar
    else:
        tau += bd.tau_max * (-pp.k_limit * over + pp.k_limit * under)
        tau -= np.where(viol, pp.c_limit * bd.tau_max * jvel * 0.1, 0.0)
    tau = np.where(hj, tau, 0.0)
    leg = _leg(bd)
    if not leg.all():       # burna tepki: burun+kafa çiftinin ataleti küçükse tork o kadar sınırlanır (sayısal sömürü önlemi)
        mh, mn = bd.mass[:, 0], bd.mass[:, N - 1]
        i_nose = (mh * mn / (mh + mn) * bd.length[:, N - 1] ** 2)[:, None]
        nr = (refidx == N - 1) & ~leg[:, None]
        tc = np.where(nr, pp.alpha_max * i_nose, np.inf)
        tau = np.clip(tau, -tc, tc)
        tot = np.sum(np.abs(tau) * nr, axis=1)        # kafaya bağlı TÜM uzuvların toplam tepki torku da sınırlı
        cap = pp.alpha_max * i_nose[:, 0]
        scale = np.where(tot > cap, cap / np.maximum(tot, 1e-9), 1.0)
        tau = np.where(nr, tau * scale[:, None], tau)

    F = np.zeros_like(st.x)
    # kemik i üzerine kuvvet çifti: +f n @ i , -f n @ parent[i], f = tau / L
    f = (tau / L)[..., None] * n
    F += f
    F -= _scatter(par, f, B, N)
    # tepki: referans kemiğe -tau (kemik 1 dahil; kafaya ayrı serbestlik yok)
    jp = refidx
    Lj = _gather(L, jp)
    nj = _gather(n, jp)
    gp = _gather(par, jp)           # referans kemiğin başlangıç noktası
    fr = np.where(hj, -tau / Lj, 0.0)[..., None] * nj
    F += _scatter(jp, fr, B, N)
    F -= _scatter(gp, fr, B, N)

    # ---- su kuvvetleri ----
    if drag:
        xp_v = _gather(st.v, par)
        for s in (0.25, 0.75):
            vs = (1 - s) * xp_v + s * st.v
            vt = np.sum(vs * u, -1, keepdims=True) * u
            vn = vs - vt
            seg = (0.5 * L * bd.width)[..., None]
            fd = -pp.rho * seg * (pp.cn * np.linalg.norm(vn, axis=-1, keepdims=True) * vn
                                  + pp.ct * np.linalg.norm(vt, axis=-1, keepdims=True) * vt)
            if pp.kn_lin or pp.kt_lin:
                fd = fd - pp.rho * seg * (pp.kn_lin * vn + pp.kt_lin * vt)
            fd = np.where(ba[..., None], fd, 0.0)
            F += s * fd
            F += _scatter(par, (1 - s) * fd, B, N)
        speed = np.linalg.norm(st.v, axis=-1, keepdims=True)
        F += -pp.rho * pp.c_point * np.sqrt(bd.mass)[..., None] * speed * st.v * bd.active[..., None]

    # ---- entegrasyon (semi-implicit Euler) ----
    inv_m = np.where(bd.active, 1.0 / bd.mass, 0.0)
    st.v += F * inv_m[..., None] * dt
    x_old = st.x.copy()
    st.x += st.v * dt

    # ---- uzunluk kısıtları (PBD, Gauss-Seidel) ----
    x = st.x
    order = list(range(1, N))
    for it in range(pp.iterations):
        seq = order if it % 2 == 0 else order[::-1]
        for i in seq:
            if not ba[:, i].any():
                continue
            p = par[:, i]
            xi = x[:, i]
            xpi = x[ar, p]
            d = xi - xpi
            dist = np.maximum(np.linalg.norm(d, axis=-1), 1e-9)
            wi = inv_m[:, i]
            wp = inv_m[ar, p]
            k = np.where(ba[:, i], (dist - bd.length[:, i]) / (dist * (wi + wp + 1e-12)), 0.0)
            corr = k[:, None] * d
            x[:, i] = xi - wi[:, None] * corr
            x[ar, p] = xpi + wp[:, None] * corr
    st.v = (x - x_old) / dt
    st.v[~bd.active] = 0.0

    st.t += dt
    st.theta, st.jvel, st.tau = theta, jvel, tau
    st.phi, st.omega = _heading(st.x, st.v, bd)


# --------------------------------------------------------------------------- #
#  Ölçüm yardımcıları (Faz 2/3'te sensör ve fitness için kullanılacak)
# --------------------------------------------------------------------------- #
def _meas_mass(bd: Bodies):
    """Ölçümlerde (kütle merkezi, hız) kullanılan kütleler: yeni modda burun (görünmez yön taşıyıcısı) hariç."""
    m = np.where(bd.active, bd.mass, 0.0)
    m[:, bd.N - 1] = m[:, bd.N - 1] * _leg(bd)
    return m


def center_of_mass(st: State, bd: Bodies):
    m = _meas_mass(bd)
    return np.sum(st.x * m[..., None], axis=1) / m.sum(1, keepdims=True)


def com_velocity(st: State, bd: Bodies):
    m = _meas_mass(bd)
    return np.sum(st.v * m[..., None], axis=1) / m.sum(1, keepdims=True)


def constraint_error(st: State, bd: Bodies):
    """Maksimum bağıl kemik uzunluğu hatası (B,)."""
    xp = _gather(st.x, bd.parent)
    L = np.linalg.norm(st.x - xp, axis=-1)
    err = np.abs(L - bd.length) / bd.length
    return np.where(bd.bone_active, err, 0.0).max(1)


def momentum(st: State, bd: Bodies):
    """Doğrusal ve açısal momentum . Sürtünmesiz testler için."""
    m = np.where(bd.active, bd.mass, 0.0)
    p = np.sum(st.v * m[..., None], axis=1)
    Lz = np.sum(m * (st.x[..., 0] * st.v[..., 1] - st.x[..., 1] * st.v[..., 0]), axis=1)
    return p, Lz


# --------------------------------------------------------------------------- #
#  Hazır test canlıları
# --------------------------------------------------------------------------- #
def chain_spec(n_seg=5, seg_len=0.5, mass=1.0, width=0.15, tau_max=3.0,
               jrange=0.9, jdamp=0.15, head_mass=2.0):
    """Balık benzeri zincir: kafa öne bakar, gövde arkaya (theta0 = pi) uzanır."""
    n = n_seg + 1
    parent = [0] + list(range(0, n - 1))
    theta0 = [0.0, np.pi] + [0.0] * (n_seg - 1)
    m = [head_mass] + [mass] * n_seg
    lo = [0.0] + [-jrange] * n_seg
    hi = [0.0] + [jrange] * n_seg
    return dict(parent=parent, length=[1.0] + [seg_len] * n_seg, width=[0.0] + [width] * n_seg,
                mass=m, theta0=theta0, jlo=lo, jhi=hi, tau_max=[0.0] + [tau_max] * n_seg,
                jdamp=[0.0] + [jdamp] * n_seg)


def star_spec(n_arms=4, arm_len=0.6, seg=2, **kw):
    """Kafadan dallanan kollu yapı: dallanmayı (bir noktaya birden çok uzuv) test eder."""
    parent, theta0, length, width, mass, lo, hi, tau, jd = [0], [0.0], [1.0], [0.0], [2.0], [0.0], [0.0], [0.0], [0.0]
    for a in range(n_arms):
        ang = 2 * np.pi * a / n_arms
        prev = 0
        for s in range(seg):
            parent.append(prev)
            prev = len(parent) - 1
            theta0.append(ang if s == 0 else 0.0)
            length.append(arm_len)
            width.append(0.15)
            mass.append(1.0)
            lo.append(-0.9)
            hi.append(0.9)
            tau.append(3.0)
            jd.append(0.15)
    return dict(parent=parent, length=length, width=width, mass=mass, theta0=theta0,
                jlo=lo, jhi=hi, tau_max=tau, jdamp=jd)
