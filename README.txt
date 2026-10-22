Evrimlesen Yuzuculer
====================
Kurulum:   pip install numpy pygame
Arayuz:    python app.py                                      (egitimi baslatir)
           python app.py --load ornekler/denizanasi.json     (dogrudan WASD ile dene)
Egitim:    python train.py --out runs/deney1 --workers 4
           python train.py --out runs/d2 --seed-from ornekler/ornek_tasarim_balik.json --freeze-body
              (tasarimdan egitim: vucut sabit, sadece beyin evrimlesir)
Dogrulama: python validate.py dosya.json    (hiz fizik adimina bagli mi?)
Analiz:    python eval_dirs.py dosya.json   (8 yonde performans + dogrulama)
Testler:   python test_physics.py / test_genome.py / test_evolution.py / test_app.py

FIZIK SURUMU 4: dogrusal (viskoz) anizotropik surtunme, dt=0.005, tork siniri + ESKLEM SINIR YAYI/TOPLAM TORK ATALETLE SINIRLI
(v3'te hafif+kisa kemik zincirlerinde sinir yayi sayisal olarak patlayip eklem torku on binlere, eklem hizi yuzlerce rad/s'e
ciktigi ve sonucun dt'ye bagli oldugu goruldu; glitch.json). Ornek canlilar v4'te yeniden dogrulandi (hizlari %5-14 dustu).
Eski surumlerle egitilmis dosyalar "[eski fizik]" etiketlidir; hizlari/davranislari degisir.
Baslangic yonu: kafa YUKARI (90 derece).
MUFREDAT (varsayilan adaptive): 0 = sadece on, 1-8 = yon araligi +-22.5..180 derece, 9 = yon degisimi.
'Komut yok = dur' (kademe 10) SIMDILIK KAPALI; acmak icin: python train.py --stop-command (app.py icin de ayni bayrak).
Sampiyon yeterince iyiyse zorluk artar, uzun sure cok kotuyse geri alinir.

TAKILMA ONLEME (v1.1):
  * Odul kapisi: enerji/donme/titreme cezalari, canli gercekten ilerlemedikce kucuk uygulanir (EvalConfig.gate_*).
    Boylece 'kipirdamamak' tek basina secilmez.
  * Takilma algilama: sampiyon fitness'inin 5 nesillik ortalamasi 15 nesil iyilesmezse mutasyon buyuklugu x1.5'ten
    x3'e kadar artar, gocmenlerin yarisi elitin agir mutasyonlu kopyasi olur. Iyilesince sifirlanir.
    Kapatmak icin: --no-stall-boost. Arayuzde 'takilma: N nesil' satiri gorunur.

ORNEKLER (ornekler/, hepsi fizik surumu 3, dogrulamadan gecti):
  ornek_simetrik_balik.json  6 uzuvlu simetrik zincir, dalga ile yuzer, 8 yon ort. 0.51 m/s
  denizanasi.json            5 uzuvlu dallanan govde, 8 yonde de ~1.2 m/s (nesil 256; v4'te ~1.1 m/s)
  yilan.json                 6 uzuvlu zincir, one 0.85 m/s ama geri/sola zayif (nesil 169)
  ornek_tasarim_balik.json   editor baslangic tasarimi (beyin rastgele)

GOVDE YONU = KAFANIN YONU (v1.3, yeni canlilar icin varsayilan):
  Kafaya bagli sabit uzunluklu, motorsuz bir 'burun' kemigi (fizikte son nokta N-1) kafanin yonunu tasir; govde acisi
  (phi) kafadan buruna giden dogrultudur. Kafaya bagli TUM uzuvlar (kemik 1 dahil) burna gore eklemlenir ve motorludur;
  tepki torku kafaya gider (ic kuvvet, momentum korunur). Boylece merkezde kafasi olan simetrik canlilar (denizanasi
  gibi) simetrik surulunce yon sapmaz. Burun GORUNMEZ (yon kafadaki siyah noktayla gosterilir) ve kutle merkezi/hiz olcumlerine dahil degildir;
  fizikte yalnizca atalet saglar (1 kg, 0.5 m, physics.NOSE_*).
  Ek sayisal onlem: burun+kafa ataletine gore tork siniri (tek uzuv ve toplam).
ESKI OMURGA MODU (opsiyonel, canli basina): kemik 1 pasif omurga, govde yonunu o belirler. Bayragi olmayan eski JSON
  dosyalari bu moda duser (davranislari birebir ayni kalir: orijinal koda gore olculdu). Editor: O tusu / panel dugmesi
  (kafaya bagli uzuvlarin acilari referans degistigi icin degisir). Egitim: python train.py --legacy-spine
  (rastgele canlilar eski modda dogar; tasarimdan egitimde mod tasarimdan gelir).

KEMIK 1 (v1.4): yeni (kafa yonu) modda kemik 1 siradan bir uzuvdur: silinebilir, aynalanabilir, mutasyonla kalkabilir;
  uzuv sayisi 1'e inebilir (mutasyon alt siniri --min-limbs ile, varsayilan 4). Eski omurga modunda kemik 1 hala korunur.

CHAPTER 2 / ASAMA 1 - EKOSISTEM (yeni):
  python eco.py                          pencereli izleme (rastgele 100 canli, sarmal 40x40 m dunya)
  python eco.py --headless --seconds 600 grafiksiz kosu; ozet satirlari + log.csv + hayatta kalanlarin JSON'u
  Testler: python test_world.py / test_eco.py
  Kurallar (world.py, WorldConfig'ten ayarlanir): sarmal dunya; yem kaynaklari sabit hizla yem uretir ve yavasca kayar;
  yem enerji verir; hareket = gercek mekanik guc x 0.01; bazal = 0.01 x kutle; enerji<=0 -> olum; algi: en yakin yemin
  yonu beynin 'hedef yon' girisine gider (menzil disi = 0). Saldiri sonraki asama.
  Not: rastgele beyinler yeme gitmez (ort. hiz ~0.02 m/s); asama 1'de yalnizca kaynak yaninda dogan canlilar yer.

CHAPTER 2 / ASAMA 2 - UREME (yeni):
  Beyin v2: 85 giris (79 + yem yakinligi, kendi enerjisi, en yakin canlinin yonu(2)/yakinligi/enerjisi), 27 cikis
  (24 motor + esesyli ureme, esesiz ureme, isirma[henuz kullanilmiyor]). v1 beyinler yukseltilince davranislari ayni kalir;
  v2 beyinler v1 kodunda (DENE) calisir. Kaydedilen eco_*.json / survivor_*.json dosyalari v2 beyinlidir.
  Esesiz: ebeveyn 80 oder, yavru 40 ile dogar. Esesli: kafalari 2 m icindeki, ikisi de isteyen cift 30'ar oder, yavru ikisinin
  capraz+mutasyonu. Olgunlasma 5 sn. Batch gerekirse genisler. --min-pop N: nufus N'in altina inerse kurtarma (varsayilan kapali).
  Rastgele baslangic beyinlerinin ureme/isirma cikis bias'i want_bias (varsayilan 0; -1.5 gibi negatif degerler ureme istegini
  tamamen bastirir, hic dogum olmaz); brain_gain ilk beyinlerin agirligini carpar.
  NOT: rastgele baslangicla yem aramanin evrimlestigi henuz gosterilmedi (README'deki olcumlere ve konusmaya bak).

TOHUMLU BASLANGIC (v2.2, VARSAYILAN): python eco.py artik ornekler/ icindeki 3 egitilmis canliyla (denizanasi, yilan, balik)
  baslar: ilk 3 birey dosyalarin aynisi, kalanlar mutasyonlu kopyalari. Eski v1 beyinler v2'ye yukseltilirken uremek ve
  isirma istekleri su bias'la baslar: esesli +1.0, esesiz +0.6 (acik), isirma -2.0 (kapali); evrim bunlari degistirebilir.
  Esesli isteyip es bulamayan canli kendi basina boler. Rastgele beyinle baslamak icin: --random (yem aramayi ogrenemeden eriyor).
  Kendi canlilarin icin: python eco.py --seed-from a.json b.json ...
  Olcum (tohum 0, 100 canli, 600 sn): nufus ~100-158'de surdu, 1501 yem yendi, 127 esesiz + 587 esesli dogum, ort. hiz ~0.2 m/s,
  ~43 nesil. Tek kosu: hangi govde planinin basardigi henuz izlenmiyor.

CHAPTER 2 / ASAMA 3 - YIRTICILIK (v2.3, AYARLANMADI):
  Her canlinin beslenme genotipi var: diet in [0,1] (0 otcul, 1 etcil). Yavruda ebeveyn(ler)in diet'i + gauss(0.05), esesli
  ise ortalamasi. Otcul en yakin YEMI, etcil en degerli/yakin CANLIYI 'hedef yon' olarak gorur (omnivor skora gore).
  Uzmanlasma bedeli: yem verimi = 1-0.7*diet, et verimi = 1-0.7*(1-diet).
  Isirma: beyin cikisi (3.) > 0.5, bekleme (3 sn) bitmis, enerji yeterli -> DENEME (5 enerji). Kafalar 0.7 m icindeyse av olur
  (kafasi yenir), saldirgan 10*av_kutlesi + 0.5*av_enerjisi kadar enerji alir (verimle carpilir). Karsilikli isirmada enerjisi
  yuksek olan kazanir, esitse ikisi de olur. Hepsi WorldConfig'ten ayarlanir (r_bite, bite_cd, cost_bite, k_meat, meat_frac,
  digest_penalty, carnivore_frac). Soy takibi: her canli tohum soyunu (lineage) tasir; panelde sayilari gorunur.
  OLCUM (tohumlu, 100 canli): etcil %20 + menzil 1.2/bekleme 1/et 20 -> 100 sn'de 230 oldurme, nufus 30'a coktu, denizanasi
  soyu yok oldu. Etcil %10 + menzil 0.7/bekleme 3/maliyet 5/et 10 (mevcut varsayilan) -> etciller ~200. sn'de tukendi,
  1500 sn'de soy dagilimi [denizanasi 0, yilan 83, balik 1]: YILAN tekeli. Yirticilik nisi henuz surdurulemedi.

EKONOMI AYARI (v2.4): yilan tekeli neden? Olcum (tohumlu, 60 canli, yirticilik kapali, 300 sn, soy sayilari deniz/yilan/balik):
  c_move 0.01 (eski): 13 / 38 / 11 (yilan one gecti, hizli ama pahali yuzuculer eriyor)
  c_move 0.003:       35 / 45 / 26 (t=200)  -> uc soy bir arada
  + 3 kaynak, sigma 2.5, oran 0.5: 35 / 54 / 12 (t=300)
  Varsayilanlar artik: c_move 0.003, 3 yem kaynagi, sigma 2.5, oran 0.5. Yirticilikla (etcil %10) t=200'de soylar 25/29/19, etcil 7->2
  (etciller yavas yavas azaliyor; surdurulebilir etcil nisi hala gosterilmedi). Kosular KISA (<=300 sn) ve tek tohum.

ETCIL NISI TARAMASI (v2.5, tohumlu 60 canli, 250 sn, etcil oranini 'etcil/otcul' olarak izle):
  P1 etcil %25: 11->4 etcil, oldurme ilk 50 sn'de 62, sonra 3'e dustu (komsular bitince av kalmiyor).
  P2 (k_meat 25, menzil 1.0, bite 3): etciller zengin (E~150) ama 11->2: bolunme 160 gerektiriyor, seyrek etcil es bulamiyor.
  P3 (+ bolunme 120): etcil patlamasi 113 etcil/15 otcul, 50 sn'de 635 oldurme = HATA: av kazanci k_meat*kutle enerjiyi yoktan
     yaratiyordu (kendi yavrusunu yiyen etcil bolunme bedelinden fazla kazaniyordu). k_meat varsayilani 0, meat_frac 0.8 yapildi
     (kazanc <= av enerjisi; test_no_cannibal_farm).
  P5 (enerji korunumlu et, bolunme 120): etciller 1'e dustu, nufus 44'e indi -> korunumlu et tek basina surdurulebilir etcil nisi
     vermiyor (av enerjisi ~40, basarisiz deneme maliyeti, avci/av yogunlugu).

LES SISTEMI (v2.6, enerji korunumlu):
  Dogumda ebeveynin kaybettigi enerjinin body_frac'i (0.5) canlinin 'vucut enerjisi' olur. Olunce (aclik/isirik) vucut enerjisi
  (+ isirikta avin artan enerjisi) LES olur, 60 sn sonra curur. Etciller canli avi da lesi de yer (les verimi 1-0.7*(1-diet)).
  Isirik kazanci = meat_frac(0.8) * (av enerjisi + vucut enerjisi) * verim. Havaya isirmak BEDAVA (menzilde canli yoksa isirik
  olayi yok; once her deneme 5 enerji tutuyordu ve etciller 90 sn'de aclıktan oluyordu). World.balance_residual() ~0 olmali
  (test_world.py).
  OLCUM (tohumlu 60 canli, etcil %25, 250 sn): varsayilan: etciller yasiyor ama 10->4 (bolunme 160'a ulasamiyorlar, E~158);
  bolunme 120 + eslesme menzili 4 m: 16->9 etcil, nufus 86->42. Surdurulebilir denge HENUZ YOK; kosular kisa, tek tohum.
