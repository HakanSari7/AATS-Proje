import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

# --- YARDIMCI: TÜRKÇE KARAKTER DÜZELTME ---
def tr_lower(text):
    return text.replace('I', 'ı').replace('İ', 'i').lower()

# -------------------------------
# 1. ULTRA GELİŞMİŞ VE BÜYÜTÜLMÜŞ VERİ SETİ (Yaklaşık 700 Kayıt)
# -------------------------------
raw_data = {
    "Elektrik": [
        "Sokak lambası yanmıyor", "Gece sokak zifiri karanlık", "Direk üzerimize devrilecek gibi duruyor",
        "Lamba sürekli yanıp sönüyor, disko gibi", "Aydınlatma yetersiz, önümüzü göremiyoruz", "Sokak lambasının camı kırık",
        "Direkten kıvılcımlar çıkıyor", "Lamba gün boyu sönmüyor, israf", "Mahallemiz çok karanlık korkuyoruz",
        "Parktaki ışıklar çalışmıyor", "Elektrik direği yan yatmış", "Sokak lambası pır pır ediyor",
        "Geceleri burası çok tekin değil, ışık lazım", "Direk paslanmış yıkılacak", "Lamba patlamış değişmesi lazım",
        "Sokak aydınlatması komple kesik", "Direklerin kabloları dışarıda tehlikeli", "Karanlıktan dolayı hırsızlık oluyor",
        "Lamba çok cılız yanıyor", "Aydınlatma direğine araç çarptı", "Sokak girişindeki lamba sönük",
        "Direk dibindeki sigorta kapağı açık", "Mahalle zifiri karanlık göz gözü görmüyor", "Lambalar 3 gündür yanmıyor",
        "Sokak lambası ağacın içinde kalmış aydınlatmıyor", "Direk çok eski sallanıyor", "Aydınlatma lambası yola düşmüş",
        "Lambanın başlığı kopmak üzere", "Sokaktaki tüm lambalar sönük", "Direkten cızırtı sesleri geliyor",
        "Lamba bir yanıp bir sönüyor", "Sokağın başındaki ışık yanmıyor", "Parkın aydınlatması kırılmış",
        "Direk yan yatmış tehlikeli", "Lamba gündüzleri de yanıyor", "Gece sokağa çıkmaya korkuyoruz karanlık",
        "Direğin dibinde kablolar açıkta oynuyorlar", "Lamba patlak", "Sokak aydınlatma direği araca çarpmış",
        "Lamba yetersiz kalıyor", "Sokak lambasi yanmiyor", "Her yer karanlik", "Lamba yanip sonuyor",
        "Aydinlatma bozuk", "Direk yan yatmis", "Lamba gunduz de yaniyor", "Sokak isiklari yanmiyor",
        "Lamba patlak", "Direk lambasi kirik", "Gece onumuzu goremiyoruz", "Parkin isiklari sonuk", "Ampul patlamis",
        "Elektrik telleri kopmuş yerde duruyor", "Trafo patladı duman çıkıyor", "Elektrik panosunun kapağı açık tehlikeli",
        "Kablolar sarkmış", "Elektrik direğinden kıvılcım çıkıyor", "Mahallede elektrik kesintisi var",
        "Trafo çok ses yapıyor", "Elektrik kutusu yanmış", "Yüksek gerilim hattı sarktığı için tehlike var",
        "Sigorta kutusu patladı", "Elektrik telleri ağaçlara değiyor", "Yer altı kablosu açığa çıkmış",
        "Elektrik panosu devrilmiş", "Direk dibinde kaçak var", "Elektrik sayaç kutusu kırık", "Teller birbirine değiyor",
        "Evdeki voltaj sürekli dalgalanıyor", "Teller birbirine çarpıp şerare yapıyor", "Sokaktaki kutudan cızırtı sesi geliyor",
        "Tedaş elektrikleri kesti", "Mahallenin yarısında elektrik var yarısında yok", "Elektrik direğinin sigortası attı",
        "Voltaj düşüklüğünden beyaz eşyalar bozuldu", "Sokaktaki elektrik kutusuna araba çarptı", "Kablo kopmuş yolda kıvranıyor elektrik olabilir",
        "Direkten ateş çıkıyor itfaiye lazım mı?", "Elektrik sayacının kapağı yok", "Trafo binasının kapısı açık çocuklar giriyor",
        "Rüzgarda teller birbirine değiyor ışıklar gidip geliyor", "Ana hat kablosu çok sarkmış kamyon takılacak", "Yer altı kablosu kazı yaparken koptu",
        "Elektrik direği paslanmış devrilmek üzere", "Mahallede faz eksikliği var sanayi elektriği yok", "Elektrik teli kopmus",
        "Trafo patladi", "Pano kapagi acik", "Kablolar sarkiyor", "Direkten kivilcim cikiyor", "Elektrikler kesik",
        "Trafo duman atiyor", "Elektrik kutusu yanmis", "Sigorta atti", "Teller agaca degiyor", "Yeralti kablosu kopuk",
        "Direkte kacak var", "Voltaj dusuklugu var", "Tellerden ses geliyor", "Sarter atti", "Ceryanlar kesildi", "Kablo koptu",
        # --- YENİ EKLENEN VARYASYONLAR ---
        "Cereyan yok sabahtan beri", "Şartel sürekli atıyor", "Elektrikler gitti", "Sokak zifiri karanlık oldu",
        "Trafodan bomba gibi ses geldi", "Direkteki lamba pat pat ötüyor", "Elektrik panosundan koku geliyor",
        "Kablolar yere değiyor çarpılacağız", "Sokak aydınlatmaları neden yanmıyor?", "Işıklar yandı söndü televizyon yandı",
        "Karanlıkta kaldık ışıklar yok", "Direğin tepesinden ateş damlıyor", "Elektirik direği devrildi",
        "Aydınlatma direkleri gündüz boşuna yanıyor", "Gece lamba yanmıyor, gündüz yanıyor", "Kablo kopuk yolda duruyor",
        "Şebeke elektriği dalgalı geliyor", "Sokaktaki sigorta kutusu parçalanmış", "Trafonun etrafı açık tehlike saçıyor",
        "Fazın biri yok", "Tedaş ne zaman yapacak elektriği", "Sokak lambamiz hic yanmiyor"
    ],
    "Yol": [
        "Sokakta derin çukurlar oluştu", "Asfalt paramparça oldu", "Yolda göçük var", "Kaldırım taşları yerinden oynamış",
        "Yaya geçidi çizgileri silinmiş", "Araçla geçerken altı vuruyor", "Yol köstebek yuvasına döndü",
        "Parke taşları sökülmüş, yürümek imkansız", "Engelli rampası yok", "Kaldırım işgal edilmiş, yürüyemiyoruz",
        "Asfalt yaması çöktü", "Yol kenarındaki bordür taşları kırık", "Sokak toz toprak içinde, asfaltlanmalı",
        "Mazgal kapağı çukurda kalmış teker düşüyor", "Yolda tümsek var", "Arabamın ön takımı dağıldı bu yol yüzünden",
        "Belediye ne zaman asfalt dökecek?", "Çukurlardan kaçacağız diye kaza yapacağız", "Her yağmurda yol çamur deryası oluyor",
        "Kaldırımda yürürken ayağım burkuldu", "Taşlar oynuyor üstümüz başımız çamur oldu", "Burası tarla yolu gibi olmuş",
        "Sokak ortasında kuyu gibi delik var", "Yol çalışması yarım bırakıldı", "Asfalt erimiş vıcık vıcık",
        "Yoldaki kasis çok yüksek, altını vuruyoruz", "Kaldırım çok dar bebek arabası geçmiyor", "Yol çökmüş tehlike arz ediyor",
        "Sokak girişindeki parkeler kırık", "Yolun ortasında inşaat artıkları var", "Asfalt delik deşik",
        "Yolun çizgileri hiç görünmüyor", "Rögar kapağı yol seviyesinden çok aşağıda", "Kaldırım taşları kırık dökük",
        "Yolda mıcır var kayıyoruz", "Sokak araları çok bozuk", "Ana caddede asfalt kabarmış",
        "Yol kenarı çökmeye başladı", "Kaldırımın ortasında direk var geçilmiyor", "Bozuk yollar yüzünden lastiğim yarıldı",
        "Sokak çok dar araçlar geçemiyor", "Asfalt yaması çok kötü yapılmış", "Kaldırım taşları yerinden çıkmış",
        "Yol kenarındaki su gideri yolu bozmuş", "Stabilize yol çok toz yapıyor", "Yolun eğimi yanlış su birikiyor",
        "Sokak girişinde büyük bir hendek var", "Kaldırımda görme engelli yolu yok", "Yolun ortası çökmüş",
        "Sokaktaki parkeler yerinden oynamış ses yapıyor", "Yolun ortasında tümsek oluşmuş altını vuruyoruz",
        "Logar kapağı yol seviyesinden düşük çukura düşüyoruz", "Kilit taşları yerinden çıkmış ayağımız takılıyor",
        "Yağmurda burası gölet oluyor geçemiyoruz", "Sokak asfaltlanmadı hala toprak yol",
        "Bebek arabasıyla kaldırımdan gidemiyoruz taşlar bozuk", "Yolun kenarı çökmüş araba düşecek", "Asfaltın yarısı var yarısı yok",
        "Kasis boyanmamış gece görmedik uçtuk", "Kaldırımda yürümek cambazlık istiyor", "Yolun eğimi ters eve su giriyor",
        "Sokakta çalışma bitti ama çukurları kapatmadılar", "Asfalt yama yama oldu yine bozuldu",
        "Yolun parkeleri oynuyor ses yapıyor uyuyamıyoruz", "Yolda yarık var lastik girdi çıkamadı",
        "Okul önündeki yol çok bozuk çocuklar düşüyor", "Kaldırım yüksekliği çok fazla inip çıkamıyoruz", "Tekerlekli sandalye rampası kırılmış",
        "Yolda cukur var", "Asvalt cok kotu", "Kaldirim taslari sokulmus", "Yol bozuk araba gitmiyor", "Sokak delik desik",
        "Yolda gocuk olustu", "Arabanin alti vurdu", "Yollar kostebek yuvasi", "Parke tasi kirik", "Engelli rampasi yapilmamis",
        "Yol cok tozuyor", "Asfalt erimis", "Yol cokmus", "Kaldirimda yuruyemiyoruz", "Yolun cizgileri silinmis",
        "Kasis cok yuksek altini vurduk", "Yolda micir var", "Sokak camur icinde", "Asvalt yamasi coktu", "Bordur tasi kirik",
        "Yollar berbat", "Cukura dustuk",
        # --- YENİ EKLENEN VARYASYONLAR ---
        "Mersin sıcağında asfalt eridi lastiğe yapışıyor", "Kaldırımlar yürünmez halde", "Yol çalışması sonrası çukuru kapatmadılar",
        "Asfaltta devasa bir delik açıldı", "Yol göçtü araba düştü düşecek", "Sokağı kazdılar öylece bıraktılar",
        "Kasis o kadar yüksek ki arabayı parçalayacak", "Mıcır döküp gitmişler her yer toz", "Kaldırım taşı koptu ayağım takıldı",
        "Sokak köstebek yuvasına döndü yama lazım", "Asvaltta çatlaklar var", "Bordürler yerinden fırlamış",
        "Yolda kocaman hendek var", "Arnavut kaldırımı bozulmuş sökülüyor", "Yol hizası bozuk su birikintisi kalıyor",
        "Engelli yolu işgal edilmiş ve kırık", "Parke taşı yerinden fırladı", "Yağmurda çamurdan sokağa girilmiyor",
        "Duble yolda çökme var", "Yamalar hep sökülmüş yollar bozuk", "Sokak asfaltsiz toprak kaldi"
    ],
    "Su": [
        "Su borusu patladı sokak göle döndü", "Şebeke suyu kesildi", "Musluktan çamurlu su akıyor",
        "Sular tazyiksiz ip gibi akıyor", "Sokaktaki vanadan su fışkırıyor", "Su sayacı patladı",
        "Temiz su borusunda sızıntı var", "Mahallede 2 gündür su yok", "Su borusu çatlamış boşa akıyor",
        "İçme suyu hattı arızalı", "Sokak ortasından kaynak suyu gibi su çıkıyor", "Su saati arızalandı",
        "Boru patlağı var acil ekip gelsin", "Sular ne zaman gelecek?", "Ana şebeke borusu delinmiş",
        "Su basıncı çok düşük", "Musluklardan paslı su geliyor", "Su borusu sızdırıyor asfaltı kaldırmış",
        "Abone hattında kaçak var", "Vana bozuldu su kapanmıyor", "Sokak su içinde kaldı boru patlak",
        "İçme suyu borusu kırılmış", "Su saatinin camı kırık", "Vanadan su damlıyor",
        "Sular çok pis akıyor içilmiyor", "Mahallede su kesintisi var", "Ana boru hattı patlamış",
        "Sokakta su sızıntısı var", "Su tazyiki kombiyi çalıştırmıyor", "Şebeke suyu bulanık",
        "Su borusu yüzeye çıkmış", "Vana kapağı kırık", "Su saati su kaçırıyor", "Borudan fışkıran su evlere giriyor",
        "Temiz su hattında arıza var", "Musluklardan kum geliyor", "Su saati çalınmış", "Boru patlağı yolu çökertmiş",
        "Sokak başındaki vana kapağı kırılmış su fışkırıyor", "Evde su hiç akmıyor kesinti mi var?",
        "Sular bulanık akıyor çamaşırlar kirlendi", "Su saatinden şırıl şırıl ses geliyor",
        "Yolun altından su kaynıyor boru patlamış olabilir", "Apartmanın girişi su doldu şebekeden kaçırıyor",
        "Su basıncı yüzünden kombi arızaya geçti", "Ana boru hattında çalışma vardı su gelmedi",
        "Musluktan pas akıyor resmen", "Su sayacının camı buğulanmış okunmuyor", "Vana yalama olmuş suyu kesemiyoruz",
        "Sokakta boşa akan su var yazık günah", "İçme suyu borusu paslanmış tadı kötü", "Su saati dondu patladı",
        "Hidrofor çalışmıyor şebeke basıncı yetersiz", "Su borusu patladi", "Sebeke suyu yok", "Musluktan camur akiyor",
        "Sular kesik", "Vana patlamis su fiskiriyor", "Su saati bozuk", "Su sizintisi var", "Icme suyu borusu kirik",
        "Su tazyiki yok", "Borudan su kaciriyor", "Sular ne zaman gelicek", "Su saati calinmis", "Boru patlagi var",
        "Su sayaci arizali", "Musluktan pas akiyor", "Sular camurlu", "Vana kirik",
        # --- YENİ EKLENEN VARYASYONLAR ---
        "Sularımız akmıyor iki gündür perişanız", "Boru patladı tonlarca su sokağa akıyor", "Şebekeden gelen su kokuyor",
        "Musluktan sarı su geliyor", "Ana hatta patlak var her yer su oldu", "Sular iplik gibi akıyor basınç sıfır",
        "Saat kapağından su sızdırıyor", "Temiz su hattı delindi yol çöktü", "MESKİ suları ne zaman verecek",
        "Sular kesildi haber vermediler", "Tazyik yok makine su almıyor", "Vanadan şarıl şarıl su boşa gidiyor",
        "İçme suyu borusunu kepçe kopardı", "Musluktan taş toprak geliyor", "Su borusu çatladı asfalttan su fışkırıyor",
        "Abonelik hattımda kaçak tespit ettim", "Sayaç vana dibinden su kaçırıyor", "Sular kireçli ve beyaz akıyor",
        "Sokak göle döndü temiz su patlağı", "Sular damla damla geliyor", "Evde sular kesik disarida su borusu patlak"
    ],
    "Kanalizasyon": [
        "Rögar kapağı taştı her yer pislik", "Logar tıkandı koku yapıyor", "Mahallede dayanılmaz bir lağım kokusu var",
        "Kanalizasyon geri tepiyor", "Tuvalet giderleri tıkandı", "Rögar kapağı çalınmış çukur açıkta",
        "Logardan fareler çıkıyor", "Pis su borusu patlamış", "Yağmur yağınca rögarlar taşdı", "Kanalizasyon hattı tıkalı",
        "Sokağa lağım suyu akıyor", "Foseptik çukuru doldu", "Mazgallar tıkalı su gitmiyor", "Banyo giderinden koku geliyor",
        "Altyapı yetersiz sürekli tıkanıyor", "Vidanjör lazım acil", "Kanalizasyon bacası kırılmış",
        "Logar kapağı ses yapıyor uyuyamıyoruz", "Pis sular yola yayılıyor", "Rögarın içi dolu temizlenmesi lazım",
        "Lağım kokusundan pencereleri açamıyoruz", "Logar kapağı çökmüş", "Kanalizasyon suyu bodrumu bastı", "Giderler çekmiyor",
        "Rögar tıkanıklığı var acil", "Pis su hattı patlamış", "Logar kapağı açık düşebilirler", "Kanalizasyon bacası yükseltilmeli",
        "Sokakta atık su birikintisi var", "Tuvalet taşması var", "Mazgalın içi çöp dolu", "Kanalizasyon borusu kırık",
        "Rögar etrafında çökme var", "Lağım suyu caddeye akıyor", "Koku sorunu var logar temizlenmeli", "Yağmur suyu gideri tıkalı",
        "Mutfak giderinden lağım kokusu geliyor", "Bodrum katı pis su bastı", "Rögar kapağı ses yapıyor araç geçtikçe",
        "Kanalizasyon borusu tıkalı su gitmiyor", "Sokaktaki mazgal yaprakla dolmuş su birikiyor",
        "Logar kapağı çökmüş içine düşeceğiz", "Bina önündeki foseptik taştı acil vidanjör",
        "Kanalizasyon hattı yetersiz sürekli tıkanıyor", "Tuvalet taşması var acil yardım", "Logar kapağı açık kalmış tehlikeli",
        "Yağmur suyu gideri tıkalı sokağı sel aldı", "Kanalizasyon bacasından koku yayılıyor",
        "Lağım fareleri cirit atıyor logar ilaçlansın", "Atık su borusu patlamış sokağa akıyor", "Logar tasti",
        "Lagim kokusu var", "Kanalizasyon tikanmis", "Tuvalet tasti", "Rogar kapagi yok", "Pis su borusu kirik",
        "Yagmur suyu gitmiyor", "Foseptik dolmus", "Mazgal tikali", "Giderden koku geliyor", "Vidanjör gonderin",
        "Logar kapagi kirik", "Sokaga pis su akiyor", "Kanalizasyon bacasi cokmus", "Lagim faresi var",
        "Kokudan durulmuyor", "Gider tikali",
        # --- YENİ EKLENEN VARYASYONLAR ---
        "Evleri lağım bastı acil yetişin", "Logar tıkalı sokak pislik içinde", "Giderlerden iğrenç bir koku yayılıyor",
        "Foseptik kuyusu doldu çekilmesi lazım", "Şiddetli yağmurda logar suyu geri tepti", "Tuvalet taşıyor gitmiyor",
        "Sokaktaki mazgal ağzına kadar çamur dolu", "Rögar kapağı üzerinden araç geçtikçe küt küt vuruyor",
        "Kanalizasyon şebekesi çöktü", "Bodrumu pis su bastı eşyalar mahvoldu", "Ana lağım hattı patlamış sokağa akıyor",
        "Logarlar çekmiyor yağmur suyu gölet oldu", "Pis su bacası tıkalı", "Gider borusu kırılmış etrafa sızıyor",
        "Koku mahalleyi sardı rögarı temizleyin", "Mazgal demirlerini çalmışlar çukur açık", "Lavabo giderleri yavaş çekiyor kanal tıkalı",
        "Kanalizasyonda tıkanıklık var vidanjör istiyoruz", "Atık sular asfalttan kaynıyor", "Logardan böcek ve sinek fışkırıyor",
        "Lagim borusu catlamis"
    ],
    "Çöp": [
        "Çöp konteyneri doldu taştı", "Sokakta çöp dağları oluştu", "Konteyner devrilmiş çöpler yola saçılmış", "Çöp arabası gelmedi",
        "Konteyner çok kötü kokuyor, yıkanmalı", "Mahallede yeterli çöp kutusu yok", "Konteynerin tekerleği kırık",
        "Çöpler günlerdir alınmıyor", "Sokak süpürülmemiş her yer pislik", "Konteyner yanmış duman çıkıyor",
        "Çöp kovasının kapağı yok", "Yere atılan molozlar alınmamış", "Eski eşyalar çöp kenarına bırakılmış",
        "Konteynerin altı delik su akıtıyor", "Çöp toplama saatleri düzensiz", "Sinek yapıyor ilaçlama lazım",
        "Geri dönüşüm kutusu dolu", "Konteyner yolu kapatıyor", "Çöp kamyonu konteyneri ezmiş",
        "Sokakta kedi köpek çöpleri dağıtıyor", "Çöp kutusu devrilmiş", "Konteynerin pedalı bozuk",
        "İnşaat molozları sokağa atılmış", "Konteyner yerinde yok", "Çöpler etrafa saçılmış", "Konteyner çok eski paslanmış",
        "Çöp suyu sokağa akıyor", "Konteyner ağzına kadar dolu", "Sokak temizliği yapılmıyor", "Çöp kutusu yanmış",
        "Konteynerin yeri değiştirilsin", "Çöpler zamanında alınmıyor", "Çöp konteynerinin pedalı bozuk açılmıyor",
        "Konteynerin altı delik pis su akıtıyor", "Sokakta başıboş çöpler rüzgarda uçuşuyor", "İnşaat molozlarını kaldırıma dökmüşler",
        "Konteyner yerinde yok çalınmış mı?", "Çöp arabası konteyneri ezmiş yamulmuş", "Pazar yeri toplandı ama çöpler duruyor",
        "Çöp kutusu çok eski paslanmış", "Yeraltı çöp konteyneri bozuldu çöp atamıyoruz", "Sokakta kedi köpek çöpleri dağıtmış",
        "Çöp kamyonu geçerken çöp suyunu yola akıttı", "Konteynerin yeri değiştirilsin koku yapıyor",
        "Eski koltuk kanepeyi çöpün yanına atmışlar", "Cop konteyneri dolu", "Copler alinmadi", "Konteyner devrilmis",
        "Cop arabasi gelmiyor", "Copler kokuyor", "Konteyner kirik", "Sokak pislik icinde", "Cop kutusu yanmis",
        "Molozlar duruyor", "Geri donusum kutusu dolu", "Cop suyu akiyor", "Sinek yapiyor", "Cop kamyonu gelmedi",
        "Cop kovasi yok", "Copler tasmis",
        # --- YENİ EKLENEN VARYASYONLAR ---
        "Sokağı çöpler götürüyor", "Temizlik işleri ne iş yapıyor çöp dağ oldu", "Konteynır ağzına kadar dolmuş kokuyor",
        "Çöp arabası 3 gündür uğramıyor", "Çöp suları yola akmış leş gibi kokuyor", "Kaldırıma moloz ve hafriyat dökmüşler",
        "Geri dönüşüm kutusunun kilidi kırık", "Çöp bidonu devrilmiş çöpler rüzgarda uçuyor", "Sokakta temizlik yapılmıyor süpüren yok",
        "Eski yatak yorganı sokağa atmışlar alın", "Konteynerin tekeri kopmuş devriliyor", "Çöpler sinek ve haşere yaptı",
        "Parkın çöp kovaları taşıyor", "Pazar artıkları dünden beri yolda", "İnşaat atıklarını çöpün yanına yığmışlar",
        "Konteyner pas içinde altı delinmiş", "Yeraltı çöp konteyneri sıkışmış açılmıyor", "Çöp kamyonu sabah çok ses yapıyor",
        "Konteyniri yanlış yere koymuşlar yolu daraltıyor", "Vatandaş çöpü poşetsiz atmış her yer pislik", "Cop tenekesi delik"
    ],
    "Park/Bahçe": [
        "Parktaki otlar insan boyu oldu", "Ağaç dalları tellere değiyor", "Çocuk parkındaki salıncak kırık",
        "Banklar parçalanmış oturulmuyor", "Parktaki çeşme bozuk", "Ağaç devrildi yolu kapattı",
        "Yeşil alanlar kurudu sulama yok", "Parkın kapısı kırık", "Kaydırak delinmiş çocuklar için tehlikeli",
        "Ağaç budaması yapılmalı", "Parktaki çöpler toplanmamış", "Spor aletleri paslanmış çalışmıyor",
        "Peyzaj düzenlemesi bozulmuş", "Ağaç kökleri kaldırımı kaldırmış", "Parkın ışıkları yanmıyor", "Havuzun suyu çok pis",
        "Oyun grubunun zemini kalkmış", "Yabani otlar temizlenmeli", "Ağaçtan düşen dallar tehlike yaratıyor",
        "Park köpek dolu giremiyoruz", "Çimler çok uzamış", "Ağaç kurudu devrilebilir",
        "Parktaki bankların tahtaları sökülmüş", "Oyun parkı zemini bozuk", "Fıskiye çalışmıyor", "Ağaçlar yola sarkmış",
        "Parkın çitleri kırık", "Çocuk oyun grubu çok eski", "Parktaki aydınlatma yetersiz", "Ağaç ilaçlaması yapılmalı",
        "Yürüyüş yolu bozuk", "Parkın içi çok pis", "Tahterevalli kırık", "Ağaç dalları penceremize giriyor",
        "Süs havuzu çalışmıyor", "Parktaki kamelya yanmış", "Refüjdeki çiçekler kurumuş", "Peyzaj sulaması yapılmıyor",
        "Parkın içindeki yürüyüş yolu çamur içinde", "Çimler biçilmediği için böcek yapıyor",
        "Oyun parkındaki kauçuk zemin kalkmış çocuklar takılıyor", "Ağacın dalı kırıldı kafamıza düşecek",
        "Parktaki spor aletlerinin pedalı kopmuş", "Kamelyanın çatısı uçmuş", "Süs havuzu yosun tutmuş kokuyor",
        "Parkın etrafındaki teller sökülmüş", "Ağaçlar çok sıklaştı budama istiyor", "Parkın sulama sistemi bozuk her yer su içinde",
        "Çocuk parkında kum havuzu çok pis", "Bankların boyası kalkmış üstümüz kirleniyor", "Refüjdeki güller kurumuş bakım lazım",
        "Meyandaki süs bitkileri sökülmüş", "Ağaç yola devrilmek üzere acil bakın", "Parkın girişi su dolmuş",
        "Parktaki agac devrildi", "Cimler cok uzamis", "Salincak kirilmis", "Banklar kirik oturulmuyor", "Agac budamasi lazim",
        "Yesil alan kurudu", "Parkin kapisi yok", "Kaydirak kirik", "Agac dallari tellere degiyor", "Spor aletleri bozuk",
        "Peyzaj kotu", "Agac kokleri yolu bozmus", "Park cok karanlik", "Havuz pislik icinde", "Yabani otlar sarmis",
        "Agactan dal dustu", "Cimenler bicilecek", "Fiskiye bozuk", "Agaclar yola sarkiyor", "Parkin citleri yikilmis",
        "Agac ilaclama", "Parktaki cicekler solmus", "Kamelya kirik", "Sus havuzu calismiyor",
        # --- YENİ EKLENEN VARYASYONLAR ---
        "Ağacın dalları balkonuma kadar girdi", "Çimler orman gibi oldu biçilmesi lazım", "Oyun parkındaki plastikler çatlamış",
        "Bankların tahtalarını kırmışlar", "Parkın içindeki otomatik sulama patlamış hep su fışkırtıyor", "Sokaktaki ağaç kurudu tehlike arz ediyor",
        "Refüjdeki çiçekler bakımsızlıktan öldü", "Parktaki fitness aletlerinin zinciri kopuk", "Tahterevallinin oturağı düşmüş",
        "Fırtınada ağaç kökünden söküldü", "Çam ağaçlarında tırtıl var ilaçlama istiyoruz", "Parktaki piknik masaları tahrip edilmiş",
        "Süs havuzunun fıskiyesi kırık suyu dışarı atıyor", "Otları temizlemedikleri için yılan çıkıyor", "Kauçuk zemin sökülmüş çocuklar düşüp yaralanıyor",
        "Park çitleri ezilmiş motorlar parka giriyor", "Budanmayan dallar elektrik tellerine sürtünüyor", "Gül ağaçları kurudu peyzaj çok kötü",
        "Parkın yürüyüş bandı çamurdan görünmüyor", "Ağaç yaprakları tüm mazgalları tıkadı", "Salincagin zinciri kopmus"
    ],
    "Trafik": [
        "Trafik lambası çalışmıyor", "Dur levhası devrilmiş", "Yol çizgileri silinmiş şerit belli değil", "Trafik aynası kırık",
        "Sinyalizasyon hatası var", "Trafik ışığı sürekli kırmızı yanıyor", "Yön tabelası ters dönmüş", "Okul geçidi levhası yok",
        "Trafik ışıkları çok sönük", "Hız sınırı tabelası kayıp", "Kavşakta ayna yok kaza oluyor", "Yaya geçidi çizgileri boyanmalı",
        "Trafik lambasının şapkası düşmüş", "Sinyalizasyon direği eğilmiş", "Girilmez levhası sökülmüş", "Trafik ışığı direği devrilmiş",
        "Okul çıkışında trafik kilitleniyor", "Trafik lambasının süresi çok kısa", "Durak tabelası devrilmiş",
        "Yaya butonuna basıyoruz çalışmıyor", "Kavşaktaki ışıklar bozuk herkes birbirine giriyor", "Sokak girişine park yapılmaz tabelası lazım",
        "Hız kesici kasisin boyası silinmiş gece görünmüyor", "Trafik levhası ağacın arkasında kalmış okunmuyor",
        "Yaya geçidi tabelası yamulmuş", "Sinyalizasyon direğinin kapağı açık", "Dönel kavşakta yön levhası yok",
        "Trafik aynası buğulanmış göstermiyor", "Radar tabelası devrilmiş", "Tek yön tabelası ters dönmüş",
        "Otobüs durağının camı kırık", "Trafik ışıkları elektrik kesilince çalışmıyor", "Refüj başındaki uyarı levhası yıkılmış",
        "Trafik lambasi bozuk", "Dur tabelasi yok", "Yol cizgileri silinmis", "Trafik aynasi kirik", "Sinyalizasyon calismiyor",
        "Isiklar surekli kirmizi", "Yon tabelasi ters", "Okul gecidi yok", "Hiz siniri tabelasi dusmus", "Kavsakta ayna yok",
        "Yaya gecidi boyanmali", "Isiklarin sapkasi yok", "Girilmez tabelasi yok", "Trafik isigi devrilmis", "Yaya butonu bozuk",
        "Trafik kilitlendi", "Levha yamulmus",
        # --- YENİ EKLENEN VARYASYONLAR ---
        "Işıklar yanmıyor kavşak felç", "Kör noktaya trafik aynası takılması şart", "Yaya geçidi çizgileri görünmüyor boyanmalı",
        "Sinyalizasyon sistemi çökmüş sarı yanıp sönüyor", "Kavşaktaki yön levhasına kamyon çarptı yamuldu", "Hız kesici tümsek belli olmuyor",
        "Kırmızı ışıkta yeşil de aynı anda yanıyor", "Otobüs durağının tabelası sökülmüş", "Sokağa ters yön tabelası takılmalı",
        "Okul bölgesi uyarı levhası ağaçtan görünmüyor", "Trafik lambası direği dipten çürümüş sallanıyor", "Yayalar için yeşil ışık çok kısa yanıyor",
        "Engelli park yeri tabelası çalınmış", "Yola çizilen hız sınırı yazısı silinmiş", "Kavşakta araçlar birbirini görmüyor ayna lazım",
        "Park edilmez tabelasını yere atmışlar", "Dur ikaz levhası ters dönmüş", "Kasisin fosforlu boyaları aşınmış gece uçuyoruz",
        "Akıllı kavşak sistemi hata veriyor", "Yön gösteren tabelalar çok karmaşık ve silik", "Trafik isiklarinda sorun var"
    ]
}

# -------------------------------
# 2. VERİ HAZIRLAMA
# -------------------------------
print("🔄 Büyük veri seti işleniyor...")
rows = []
for tur, aciklamalar in raw_data.items():
    for aciklama in aciklamalar:
        rows.append({"aciklama": aciklama, "tur": tur})

data = pd.DataFrame(rows)
print(f"✅ Toplam Eğitim Verisi: {len(data)} adet")

# -------------------------------
# 3. MODEL EĞİTİMİ (GÜNCELLENDİ)
# -------------------------------
print("\n🧠 Yapay Zeka eğitiliyor (TF-IDF & Naive Bayes)...")

# Verinin %15'i ile modelimizi test edeceğiz.
X_train, X_test, y_train, y_test = train_test_split(
    data["aciklama"], data["tur"], test_size=0.15, random_state=42
)

# YENİLİK: TfidfVectorizer eklendi. Kelimelerin frekansı (Count) yerine ağırlığı (TF-IDF) hesaplanıyor.
# sublinear_tf=True: Kelime çok tekrarlanırsa aşırı ağırlık almasını engeller.
# alpha=0.3: Hiç görmediği kelimelerde hata vermemesi için smoothing ayarı yapıldı.
model = make_pipeline(
    TfidfVectorizer(preprocessor=tr_lower, ngram_range=(1, 2), sublinear_tf=True),
    MultinomialNB(alpha=0.3)
)
model.fit(X_train, y_train)

# -------------------------------
# 4. TEST SONUÇLARI VE METRİKLER (GÜNCELLENDİ)
# -------------------------------
accuracy = model.score(X_test, y_test)
print(f"🎯 Model Genel Başarısı (Accuracy): %{accuracy * 100:.2f}")

# YENİLİK: Her bir kategori için detaylı başarı raporunu (Precision, Recall, F1-Score) yazdırıyoruz.
print("\n📊 Model Detaylı Başarı Raporu (Classification Report):")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Simülasyon Testi
testler = [
    "Sokak çok karanlık gözükmüyor",
    "Rögar kokusu mahalleyi sardı",
    "Arabamın altı vurdu çukur var",
    "Parktaki salıncak kopmuş",
    "Musluktan çamur akıyor"
]

print("\n🧪 Canlı Test Simülasyonu:")
for t in testler:
    sonuc = model.predict([t])[0]
    print(f"📝 '{t}' \n   👉 Tahmin: {sonuc}")

# -------------------------------
# 5. KAYDET
# -------------------------------
joblib.dump(model, "text_model.pkl")
print("\n💾 text_model.pkl başarıyla güncellendi.")
print("⚠️ UNUTMA: Değişikliğin etkili olması için main.py (app.py) kapatıp tekrar açman gerekiyor!")