# 🧠 TTCS - Hệ Khuyến Nghị Tuần Tự và Dự Đoán Mua Hàng Tiki (Sequential Recommendation)

Dự án này tập trung nghiên cứu và phát triển các **Hệ khuyến nghị tuần tự (Sequential Recommendation)** và **Mô hình dự đoán xác suất mua hàng (Sequential Purchase Prediction)** dựa trên chuỗi lịch sử hành vi (View, Click, AddCart, Search, Wishlist,...) của người dùng trên nền tảng thương mại điện tử Tiki.

---

## 📂 Mục lục
- Giới thiệu
- Kiến trúc & Mô hình
- Cấu trúc thư mục
- Hướng dẫn cài đặt
- Hướng dẫn chạy mô hình

---

## 📝 Giới thiệu

Hệ khuyến nghị tuần tự mô hình hóa sự thay đổi sở thích của người dùng theo thời gian bằng cách xử lý các chuỗi tương tác (session-based/sequence-based). Dự án này hỗ trợ 2 tác vụ chính:

1. **Khuyến nghị sản phẩm tiếp theo (Next-Item Recommendation)**: Dự đoán sản phẩm (`product_id`) và hành động tiếp theo (`action_type`) mà người dùng sẽ thực hiện dựa trên chuỗi hành vi trước đó.
2. **Dự đoán xác suất mua hàng (Purchase Probability Prediction)**: Dự đoán xác suất người dùng sẽ mua một sản phẩm cụ thể dựa trên lịch sử các hành động (nhấp chuột, xem, thêm vào giỏ,...) đối với chính sản phẩm đó.

**Công nghệ sử dụng:**
- **Ngôn ngữ**: Python 3.10
- **Framework**: PyTorch
- **Thư viện bổ trợ**: Pandas, NumPy, Scikit-learn, Matplotlib, Tqdm, Tabulate

---

## 🏗️ Cấu trúc thư mục

```text
.
├── checkpoints/              # Thư mục lưu trữ checkpoint mô hình tốt nhất (.pt)
├── data/
│   ├── sequential/           # Dữ liệu chuỗi hành vi (events CSV, NPZ splits)
│   ├── purchase_dataset.py   # Xử lý chuỗi tương tác cho mô hình dự đoán mua hàng
│   ├── sequential_dataset.py # Xử lý dữ liệu đầu vào cho mô hình tuần tự tiêu chuẩn
│   └── sequential_generator.py # Simulator tạo dữ liệu chuỗi hành vi giả lập
├── models/
│   ├── purchase_model.py     # Mô hình tuần tự phân loại mua hàng (LSTM/GRU + FC Head)
│   └── sequential_models.py  # Các mô hình tuần tự khuyến nghị (LSTM, GRU, BiLSTM)
├── train/
│   └── sequential_trainer.py # Trình huấn luyện & Đánh giá mô hình tuần tự tiêu chuẩn
├── main_purchase_seq.py      # Entry point huấn luyện mô hình dự đoán mua hàng
├── main_sequential.py        # Entry point huấn luyện mô hình khuyến nghị tuần tự
├── requirements.txt          # Danh sách các thư viện cần thiết
└── README.md
```

---

## ⚙️ Hướng dẫn cài đặt

### 1. Tạo môi trường ảo với Conda
Nên sử dụng Conda để quản lý môi trường và tránh lỗi xung đột thư viện trên Windows:
```bash
conda create -y -n ttcs_rec python=3.10
conda activate ttcs_rec
```

### 2. Cài đặt các thư viện phụ thuộc
Cài đặt các gói thư viện cơ bản thông qua Conda:
```bash
conda install -y -n ttcs_rec pandas numpy scikit-learn tqdm tabulate matplotlib
```

Cài đặt PyTorch CPU (khuyên dùng để tránh lỗi độ dài đường dẫn Windows đối với gói Conda):
```bash
conda activate ttcs_rec
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 🛠️ Hướng dẫn chạy mô hình

*Lưu ý: Nếu gặp lỗi OpenMP duplicate runtime trên Windows (`libiomp5md.dll`), hãy set biến môi trường trước khi chạy:*
- **Windows PowerShell**: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`
- **Windows Command Prompt**: `set KMP_DUPLICATE_LIB_OK=TRUE`

### 1. Chạy mô hình Khuyến nghị sản phẩm tiếp theo (Next-Item Rec)
Mô hình sử dụng mạng LSTM/GRU/BiLSTM để học biểu diễn chuỗi hành vi người dùng đối với toàn bộ các sản phẩm và dự đoán sản phẩm + hành động tiếp theo.
```bash
python main_sequential.py --model lstm --epochs 10 --batch-size 256
```
**Tham số tùy chọn:**
- `--model`: Kiến trúc mạng, chọn giữa `lstm`, `gru`, hoặc `bilstm` (mặc định: `lstm`).
- `--epochs`: Số lượng epoch huấn luyện (mặc định: `10`).
- `--batch-size`: Kích thước batch (mặc định: `256`).
- `--seq-len`: Độ dài chuỗi lịch sử đưa vào mô hình (mặc định: `10`).

### 2. Chạy mô hình Dự đoán xác suất mua hàng (Purchase Prediction)
Mô hình lấy chuỗi hành vi của người dùng đối với một sản phẩm cụ thể và dự đoán xác suất người dùng đó mua sản phẩm đó.
```bash
python main_purchase_seq.py --model lstm --epochs 15 --batch-size 128
```
**Tham số tùy chọn:**
- `--model`: Kiến trúc mạng, chọn giữa `lstm` hoặc `gru`.
- `--epochs`: Số lượng epoch huấn luyện (mặc định: `20`).
- `--batch-size`: Kích thước batch (mặc định: `128`).
- `--seq-len`: Độ dài tối đa của chuỗi hành động trên sản phẩm đó (mặc định: `10`).

Sau khi huấn luyện xong, kết quả đánh giá (Loss, Accuracy, ROC AUC, PR AUC, Precision, Recall, F1-Score) trên tập kiểm thử (Test Set) sẽ được in ra dưới dạng bảng. Đồng thời, biểu đồ huấn luyện sẽ được tự động lưu vào `data/sequential/figures/purchase_metrics.png`.
