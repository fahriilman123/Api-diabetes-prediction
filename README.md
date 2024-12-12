# Aplikasi Prediksi Penyakit Diabetes Berbasis Flask

Aplikasi ini menggunakan **Flask** untuk menyediakan API yang dapat memprediksi apakah seseorang berisiko terkena penyakit diabetes berdasarkan beberapa faktor kesehatan. Aplikasi ini mengimplementasikan **Extreme Learning Machine (ELM)** untuk melakukan prediksi.

## Deskripsi
API ini menerima data dalam format JSON dengan berbagai parameter kesehatan, seperti tekanan darah, indeks massa tubuh (BMI), kebiasaan merokok, dan lainnya. Berdasarkan data ini, model akan memprediksi apakah seseorang berisiko terkena penyakit diabetes.

### Endpoints:
- **`POST /predict`**: Endpoint ini menerima data JSON yang berisi faktor-faktor risiko diabetes dan memberikan prediksi (0 atau 1).
  
## Fitur
- **Prediksi Risiko Diabetes**: Menggunakan model **Extreme Learning Machine (ELM)** untuk prediksi.
- **Preprocessing Data**: Menggunakan **scaler** untuk menstandarisasi fitur input.
- **Cek Fitur Input**: Memastikan bahwa semua fitur yang diperlukan ada dalam permintaan.

## Instalasi

1. **Clone repository ini:**
   ```bash
   https://github.com/fahriilman123/Api-diabetes-prediction.git
