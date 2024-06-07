import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.preprocessing import StandardScaler
import joblib

# Modeli ve ölçekleyiciyi yükle
scaler = joblib.load('scaler.joblib')
model = joblib.load('heart_attack_predictor_decision_tree.joblib')

# Özelliklerin isimlerini ve hangi aralıklarda değer girmesi gerektiğini tanımla
feature_info = {
    'age': 'Yaş (25-77)',
    'sex': 'Cinsiyet',
    'cp': 'Göğüs Ağrısı Türü (0-3)',
    'trestbps': 'İstirahat Kan Basıncı (94-200)',
    'chol': 'Kolestrol (126-564)',
    'fbs': 'Açlık Kan Şekeri (0: <120, 1: >=120)',
    'restecg': 'Dinlenme Elektrokardiyografik Sonuçlar (0-2)',
    'thalach': 'Maksimum Kalp Atış Hızı (71-202)',
    'exang': 'Egzersize Bağlı Anjin (0: Hayır, 1: Evet)',
    'oldpeak': 'Egzersizle Oluşan ST Depresyonu (0.0-6.2)',
    'slope': 'Eğim (0-2)',
    'ca': 'Renkli Damalar (0-3)',
    'thal': 'Thalium Testi Sonucu (0-2)'
}

# Tahmin fonksiyonu
def predict():
    # Kullanıcının girdiği değerleri al
    user_input = []
    for feature, entry in entries.items():
        if feature != 'sex':  # 'sex' özelliğini hariç tut
            if feature == 'age':
                user_input.append(float(entry.get()))  # 'age' özelliği için dönüştürme yap
            else:
                user_input.append(float(entry.get()))  # Diğer özellikler için dönüştürme yap
    # Verileri ölçeklendir
    user_input_scaled = scaler.transform([user_input])
    
    # Tahmin yap
    prediction = model.predict(user_input_scaled)
    
    # Tahmin sonucunu kullanıcıya göster
    if prediction[0] == 0:
        messagebox.showinfo('Tahmin Sonucu', 'Kalp Krizi Riski Yok')
    else:
        messagebox.showinfo('Tahmin Sonucu', 'Kalp Krizi Riski Var')


# Ana uygulama penceresi oluştur
root = tk.Tk()
root.title('Kalp Hastalığı Tahmin Arayüzü')

# Giriş alanlarını oluştur
entries = {}
for i, (feature, info) in enumerate(feature_info.items()):
    label = tk.Label(root, text=f'{info}:')
    label.grid(row=i, column=0, padx=10, pady=5, sticky='w')
    if feature == 'sex':  # Cinsiyet özelliği için Combobox
        combo = ttk.Combobox(root, values=['Kadın', 'Erkek'])
        combo.grid(row=i, column=1, padx=10, pady=5)
        entries[feature] = combo
    else:
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[feature] = entry

# Tahmin butonunu oluştur
predict_button = tk.Button(root, text='Tahmin Et', command=predict)
predict_button.grid(row=len(feature_info), column=0, columnspan=2, padx=10, pady=10)

# Uygulamayı başlat
root.mainloop()
