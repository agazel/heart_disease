import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Veriyi yükleyin
df = pd.read_csv('heart_yeni.csv')

# Veriyi inceleyin
print(df.head(50)) #ilk 50 veri
print(df.info()) #veriler hakkında bilgi
print(df.describe()) #istatistikleri özetler

# Eksik değerleri kontrol edin
print(df.isnull().sum()) #eksik değerleri sayarak yazdırır

# Eksik değerleri doldurun
# Eksik veri oranı %50'den fazla olan sütunları değerlendirmeye almayın
threshold = len(df) * 0.5 #yüzde 50 eşik değer belirledik
df = df.loc[:, df.isnull().sum() <= threshold] #eşik değere eşit yada altında olan sütunları seçtik (aşağıda doldurmak için)

# Sayısal sütunlar için ortalama ile doldurun
numeric_columns = df.select_dtypes(include=[np.number]).columns #doldurulabilecek (sütunun sayısal olması gerekiyor ortalama kullanabilmek için) sütunların ismini belirler
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean()) #eksik verileri ortalamaları ile doldurur

# Kategorik sütunlar için mod ile doldurun
categorical_columns = df.select_dtypes(include=['object', 'category']).columns #sayısal olmayan (ortalama ile doldurulamayacak yani kategorik) sütunları seçer
for col in categorical_columns: #seçilen kategorik sütunlardaki eksik verileri de sütunda en çok tekrar eden verilerle doldurur
    df[col].fillna(df[col].mode()[0], inplace=True)

# Kategorik değişkenleri sayısal verilere dönüştürün (etiketleme işlemidir)
label_encoders = {} #sözlük oluşturduk her nesnenin etiketini saklamak için (sözlüğe kaydetme sebebimiz modeli oluşturduktan sonra sütunu tekrardan eski haline kategorik veriye dönüştürebilmemizi sağlar
# test ve eğitim veirlerinde tutarlılık sağlar , model sonuçları geldiğinde hangi sayısal ifadelerin hangi kategorik veri olduğunu görmemizi sağlar , tekrar tekrar kullanıldığında her seferinde yeniden etiketleme yapmayı önleyerek zaman ve bellek tasarrufu sağlar) 
for col in categorical_columns: #kategorik sütunları seçer
    le = LabelEncoder() #etiketleme nesnesini oluşturduk
    df[col] = le.fit_transform(df[col]) #kategorik sütunu sayısal değere dönüştürür ve günceller
    label_encoders[col] = le #kullanılan sütun etiketiyle birlikte sözlüğe eklenir

# Özellikler ve hedef değişkeni (bağımlı-bağımsız) belirleyin , veri hazrılama işlemidir (ayıklama yapılır , gereksiz verileri atarak id gibi falan modelin daha iyi çalışmasını sağlar , hedefler belirlenerek tutarlılık sağlanır)
if 'id' in df.columns: # 'id' sütunu varsa seçer
    df = df.drop(columns=['id'])          # ve kaldırır
X = df.drop(columns=['num']) # num sütunu çıkarılır çünkü x model öğrenmesi için kullanılacak yani girdiler bağımsız değişkenleri içermeli
y = df['num']  # 'num' sütununu doğrudan hedef değişken (bağımlı) olarak kullanılır o yüzden ayrı bir veri setine atılır

# 'num' sütununu 0 ve 1 değerlerine sahip olacak şekilde ayarlayın
y = y.apply(lambda x: 1 if x > 0 else 0) #sıfırdan büyükse 1 , küçükse 0 
#sınıflandırma problemi için kullanılacak , hedef değişkenin formatı düzenlendi (iki sınıf arasında ayırt etme işlemi çok daha verimli olur model eğitim açısından)   

# İşlenen verilerin son hali
print("\nİşlenen Verilerin Son Hali:")
print(df.head(50))

# Veriyi eğitim ve test setlerine bölün (stratify parametresi ile)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # %20 test , 80 eğitim -- stratify dengesiz veri setlerinde dağılımı korur ve tutarlılık sağlar

# Veriyi ölçeklendirin (her özelliğin ortalama değerini 0 ve standart sapmasını 1 yapar.)
scaler = StandardScaler() #sınıfından bir örnek oluşturulur. Bu örnek özelliklerin standartlaştırılması için kullanılacak parametreleri ve yöntemleri içerir
X_train = scaler.fit_transform(X_train) #fit eğitim verisindeki her özellik için ortalama ve standart sapmayı hesaplar
X_test = scaler.transform(X_test) #trasform hesaplananları veriye dönüştürür
# modelin eğitimi ve değerlendirmesi sırasında özelliklerin aynı ölçekte olmasını sağlar bu da modelin daha iyi performans göstermesine

# Modeli oluşturun ve eğitin
model = DecisionTreeClassifier() #karar ağacı modeli
model.fit(X_train, y_train) #fit eğitim sağlar ilki bağımsız , ikinci bağımlı değişkeni gösterir
#Karar ağacı modeli, veri setindeki özelliklerin değerlerine dayanarak bir dizi karar kuralı oluşturur ve bu kuralları kullanarak sınıflandırma yapar

# Karar ağacı şemasını çizin
plt.figure(figsize=(100,20))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No Heart Attack', 'Heart Attack'],
    filled=True,
    rounded=True,
    proportion=True,
    precision=2  # Sayısal değerlerin hassasiyetini ayarlar
)
plt.title('Karar Ağacı')
plt.show()

# Tahmin yapın
y_pred = model.predict(X_test) #x üzerinde tahminler yapar ve yprede atar
y_pred_proba = model.predict_proba(X_test)[:, 1] #tahmini olasılıkları hesaplar , 1 sadece pozitif kısmın hesaplanacağını belirtir , Bu olasılıklar her bir örnek için olumlu sınıf olma olasılıklarını içeren bir dizi olarak y_pred_proba değişkenine atanır
#Bu işlem sonucunda, eğitilmiş modelin test veri seti üzerinde yaptığı sınıf tahminleri (y_pred) ve olumlu sınıf olma olasılıkları (y_pred_proba) elde edilir. Bu tahminler daha sonra modelin performansının değerlendirilmesi veya sonuçların analizi için kullanılabilir

# Modeli değerlendirin (ne kadar tutarlı tahmin yapıyor , ne kadar iyi çalıştığını gösterir)
accuracy = accuracy_score(y_test, y_pred) #gerçek hedef değerleri (y_test) ve tahmin edilen hedef değerleri (y_pred) alarak doğruluk ölçüsünü hesaplar. Doğruluk, doğru olarak tahmin edilen örneklerin toplam örnek sayısına oranını ifade eder
roc_auc = roc_auc_score(y_test, y_pred_proba) #gerçek hedef değerleri (y_test) ve sınıf olasılıkları (y_pred_proba) alarak ROC Eğrisi Altında Alan (ROC AUC) ölçüsünü hesaplar
#ROC AUC, sınıflandırma modelinin olasılık tahminlerinin sınıflandırma performansını ölçer. Değer 0 ile 1 arasında olup, daha yüksek değerler daha iyi bir performansı gösterir.
classification_rep = classification_report(y_test, y_pred) #sınıflandırma raporu üretir. Bu rapor, sınıflandırma modelinin performansını farklı ölçümlerle (doğruluk, hassasiyet, geri çağırma, F1 skoru) detaylı olarak sunar

#Değerlendirmeleri Yazdırır
print(f'Doğruluk: {accuracy}')
print(f'ROC-AUC Skoru: {roc_auc}')
print(f'Sınıflandırma Raporu:\n{classification_rep}')

# ROC eğrisini çizin
fpr, tpr, _ = roc_curve(y_test, y_pred_proba) #yalancı pozitif oranını (FPR) ve gerçek pozitif oranını (TPR) hesaplar
roc_auc = auc(fpr, tpr) #bunları kullanarak roc auc eğrisnin grafiği çizer

#ROC grafiği yazdırma
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (alan = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Alıcı İşletim Karakteristiği (ROC)')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix (gerçek ve tahmin edilen sınıflar arasındaki ilişkiyi gösteren karışıklık matrisi elde edilir)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Konfüzyon Matrisi')
plt.show()

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba) #gerçek hedef değerleri (y_test) ve sınıf olasılıkları (y_pred_proba) alarak hassasiyet ve geri çağırma eğrilerini hesaplar
#precision, her farklı karar eşiği için hesaplanan hassasiyet değerlerini, recall ise her farklı karar eşiği için hesaplanan geri çağırma değerlerini içeren dizilerdir
#sınıflandırma modelinin performansını daha ayrıntılı olarak değerlendirmek için kullanılabilir ve özellikle karar eşiği değiştiğinde modelin performansının nasıl değiştiğini gösterir

plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall Eğrisi')
plt.xlabel('Duyarlılık (Recall)')
plt.ylabel('Hassasiyet (Precision)')
plt.title('Precision-Recall Eğrisi')
plt.legend(loc='lower left')
plt.show()

# Modelin Kaydedilmesi
joblib.dump(scaler, 'scaler.joblib') #modelin eğitiminde kullanılan ölçekleri kaydeder
joblib.dump(model, 'heart_attack_predictor_decision_tree.joblib') #modeli hangi eğitim verileriyle oluştuğuyla birlikte kaydeder

# Ülke (dataset), cinsiyet (sex) ve yaş (age) bilgilerine göre görselleştirmeler (GENEL İSTATİSTİKİ BİLGİLERİN GRAFİK HALLERİ)

# 2. Cinsiyete göre kalp krizi oranı
gender_counts = df['sex'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color='lightcoral')
plt.title('Cinsiyete Göre Kalp Krizi Oranı (%)')
plt.xlabel('Cinsiyet')
plt.ylabel('Oran (%)')
plt.xticks(ticks=[0, 1], labels=['Erkek', 'Kadın'], rotation=0)
plt.show()

# 3. Yaş dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, bins=20, color='green')
plt.title('Yaş Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Frekans')
plt.show()

# 4. Yaşa göre kalp krizi oranı
age_bins = pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80])
age_grouped = df.groupby(age_bins)['num'].mean() * 100
plt.figure(figsize=(10, 6))
age_grouped.plot(kind='bar', color='purple')
plt.title('Yaşa Göre Kalp Krizi Oranı (%)')
plt.xlabel('Yaş Grupları')
plt.ylabel('Oran (%)')
plt.show()

# 5. Cinsiyet ve yaşa göre kalp krizi oranı
plt.figure(figsize=(12, 8))
sns.boxplot(x='sex', y='age', hue='num', data=df, palette='Set3')
plt.title('Cinsiyet ve Yaşa Göre Kalp Krizi Oranı')
plt.xlabel('Cinsiyet')
plt.ylabel('Yaş')
plt.xticks(ticks=[0, 1], labels=['Erkek', 'Kadın'])
plt.legend(title='Kalp Krizi', loc='upper right', labels=['Hayır', 'Evet'])
plt.show()
