# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác

- **SEED:** 9547; chia train/val/test: 3619/775/775; SUB: 40 tin (20 spam, 20 ham).
- Danh tính học viên theo tên file báo cáo; không lặp lại thông tin cá nhân trong nội dung.
- Báo cáo này được AI hỗ trợ tổng hợp từ notebook và JSON hiện có trong thư mục nộp bài. Các con số lấy từ lần chạy có dự đoán bắt đầu lúc 15:50:05 theo đồng hồ runtime; không trộn với benchmark cũ.

## 1. Bằng chứng

| Tệp | Trạng thái kiểm tra |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) | Có output; không có output lỗi; báo cáo cuối ghi 15/15 kiểm tra đạt, đủ 10 dự đoán. Một số câu trả lời trong notebook vẫn cần bổ sung trước khi nộp. |
| [lab1_report.json](lab1_report.json) | Có seed, predictions, checks, experiments, challenge và results; khớp dấu thời gian ở output báo cáo cuối notebook. |
| [hand_calc.jpg](hand_calc.jpg) | Bản tham khảo do AI tạo với đúng dữ kiện. **Chưa đáp ứng yêu cầu ảnh bài viết tay của học viên**; cần tự làm trên giấy, chụp ảnh và thay file này trước khi nộp. |

15/15 kiểm tra tự động chỉ chứng minh các điều kiện mà bộ kiểm tra kiểm tra được; không thay thế việc hoàn thành câu hỏi, viết báo cáo, ảnh tính tay và tuân thủ quy định dùng AI.

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 (ms/tin) | USD/1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.9143 | 0.8571 | 9.384 | 7636.94 | 0.131 | 0.000000 |
| 2. BoW + MLP (deep learning) | 0.9570 | 0.8889 | 20.267 | 14284.12 | 2.408 | 0.000000 |
| 3a. Encoder (MiniLM) + kNN | 0.9000 | 0.9744 | 120.553 | 22.40 | 64.981 | 0.000000 |
| 3b. Encoder (MiniLM) + LogReg | 0.9215 | 0.9744 | 120.990 | 30.12 | 37.623 | 0.000000 |
| 3c. Encoder (MiniLM) + MLP head | 0.9206 | 0.9474 | 121.498 | 30.07 | 36.280 | 0.000000 |
| 4a. LLM nhỏ zero-shot | 0.9744 | 0.9744 | 0.000 | 1.13 | 1052.315 | 0.008830 |
| 4b. LLM lớn zero-shot | 0.9744 | 0.9744 | 0.000 | 1.07 | 1158.309 | 0.088300 |
| 4c. LLM nhỏ few-shot | 0.9744 | 0.9744 | 0.000 | 1.12 | 1661.305 | 0.019097 |

Các hàng LLM chỉ đánh giá trên SUB; dùng F1 SUB để so chất lượng giữa cả bốn cách. Chi phí API trong lab là ước tính token theo số từ, không phải hóa đơn usage thực tế. Giá và tên model được giữ theo cấu hình của bài; chưa xác minh bảng giá nhà cung cấp hiện hành. Thời gian train của embedding bao gồm encode tập train bằng encoder đóng băng.

Theo output Pareto của lần chạy này, đường Pareto gồm **BoW + LogReg, BoW + MLP, Embedding + MLP head, Embedding + LogReg**. MLP head có p95 36.280 ms, nhanh hơn LogReg head 37.623 ms nhưng F1 SUB thấp hơn (0.9474 so với 0.9744), nên hai head đều nằm trên đường Pareto. Embedding + kNN bị LogReg head lấn át: cùng F1 SUB 0.9744, nhưng kNN có p95 64.981 ms. Các hàng LLM cùng F1 SUB 0.9744 và chậm hơn LogReg head nên cũng bị lấn át theo hai trục này.

## 3. Dự đoán và thực tế

1. **ML F1:** dự đoán khoảng 0.85, precision cao hơn recall. Thực tế F1 test 0.9143, precision 1.0000, recall 0.8421. Dự đoán thấp hơn thực tế khoảng 0.0643 F1; chiều precision/recall đúng. Các từ khóa trong dữ liệu hỗ trợ BoW khá tốt, nhưng mô hình vẫn bỏ sót spam không giống mẫu phổ biến.
2. **BoW so với embedding ở 6 tin ngụy trang:** dự đoán Encoder + LogReg đúng nhiều hơn. Output thực tế: BoW đúng **5/6**, Encoder + LogReg đúng **4/6**. Tin “Your mobile number has been chosen for a complimentary holiday...” là spam: BoW đoán đúng, embedding head đoán ham. Biểu diễn ngữ nghĩa tốt không bảo đảm bộ phân loại huấn luyện trên nó sẽ tổng quát hóa tốt ở từng ví dụ; ngưỡng, dữ liệu có nhãn và cách diễn đạt mới vẫn ảnh hưởng.
3. **Model lớn và few-shot:** dự đoán model lớn tăng F1 0.02–0.05, few-shot tăng 0.01–0.03. Thực tế ba cấu hình LLM cùng F1 SUB **0.9744**; lớn đắt gấp 10 lần nhỏ zero-shot, few-shot đắt khoảng **2.16 lần**. Thêm năng lực hoặc ví dụ làm tăng chi phí nhưng không sửa được lỗi trên SUB này. Với 40 mẫu, chênh lệch nhỏ chưa đủ khẳng định hơn kém bền vững.

Quan sát thêm: MLP đạt F1 test 0.9570 so với BoW + LogReg 0.9143. Val loss thấp nhất ở chỉ số epoch 8 (epoch thứ 9 nếu đếm từ 1); sau đó train loss giảm còn val loss tăng, cho thấy overfit. Chọn checkpoint bằng validation, không chọn bằng test.

## 4. Phân tích 50 case khó

| Mô hình | F1 SUB thường | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.857 | 0.368 | 0.52 |
| BoW + MLP | 0.889 | 0.488 | 0.58 |
| Embedding + kNN | 0.974 | 0.704 | 0.68 |
| Embedding + LogReg | 0.974 | 0.612 | 0.62 |
| Embedding + MLP head | 0.947 | 0.553 | 0.58 |
| LLM nhỏ zero-shot | 0.974 | 1.000 | 1.00 |
| LLM nhỏ + prompt phòng thủ | 0.974 | 1.000 | 1.00 |

BoW + LogReg tụt nhiều nhất, khoảng **0.489 F1**. BoW + LogReg khó nhất ở thao túng/lừa đảo (12.5% đúng); BoW + MLP khó nhất ở spam giọng tự nhiên (10%). Embedding + kNN khó nhất ở nhóm nói về spam (25%), còn hai embedding head khó nhất ở spam tự nhiên (20%). Hai cấu hình LLM đúng toàn bộ các nhóm trong lần chạy này, nên không xác định được nhóm khó nhất của LLM từ 50 case này.

Hai case minh họa cơ chế lỗi của các model cổ điển:

- **Case #12, spam giọng tự nhiên:** người gửi nói bạn của mình cho số, mới đến thành phố và muốn làm quen, rồi dẫn tới trang hẹn hò. Lời chào thân thiện và ít dấu hiệu thưởng/khuyến mại khiến BoW dễ đoán ham; embedding cũng có thể đặt nó gần hội thoại cá nhân. Nội dung thực chất là tiếp cận không được yêu cầu để dụ vào dịch vụ.
- **Case #13, spam giọng tự nhiên:** thông báo ảnh sự kiện đã sẵn sàng, mời xem và đặt mua ảnh. Hình thức giống nhắc nhở giao dịch hợp lệ nên các model cổ điển có thể bỏ sót mục đích bán hàng không được yêu cầu. Cần phân biệt với một thông báo từ dịch vụ mà người dùng thực sự đã đăng ký.

LLM zero-shot và prompt phòng thủ đều đạt 100% nhóm prompt injection và mọi nhóm khác; chưa thấy cải thiện hoặc suy giảm do prompt phòng thủ trên bộ này. Không suy ra hệ thống miễn nhiễm injection. Ngoài prompt, chỉ chấp nhận nhãn hợp lệ, không cấp công cụ/quyền cho bộ phân loại, và chuyển xem xét khi các bộ phân loại bất đồng.

Bộ 50 case được viết tay và chủ đích gây khó, không phản ánh tỷ lệ tình huống ngoài đời. Dùng để tìm điểm yếu, kiểm tra hồi quy và thiết kế đánh giá bổ sung; không dùng điểm 100% để thay thế test đại diện độc lập hoặc khẳng định chất lượng triển khai.

## 5. Thử thách cá nhân

**Đề #2, SEED % 6 = 1:** đổi vocab của BoW + LogReg xuống 100 và lên 5000, train lại và báo F1 test. Giữ nguyên split, learning rate 0.5, 300 epoch và ngưỡng 0.5 để so sánh.

| Vocab | Precision | Recall | F1 test | Thời gian thí nghiệm (s) |
| --- | ---: | ---: | ---: | ---: |
| 100 | 0.9487 | 0.7789 | 0.8555 | 0.2227 |
| 2000 — baseline | 1.0000 | 0.8421 | 0.9143 | 9.384 train theo benchmark riêng |
| 5000 | 1.0000 | 0.8526 | 0.9205 | 7.8544 |

Vocab 100 bỏ mất nhiều từ ít gặp có ích cho phát hiện spam nên recall và F1 giảm. Tăng từ 2000 lên 5000 chỉ cải thiện khoảng 0.0062 F1 trong lần chia này, trong khi tăng số tham số và bộ nhớ. Không kết luận thời gian train tăng/giảm từ các số trên: hai dòng thử thách đo cả vector hóa/đánh giá và chạy ở thời điểm khác benchmark baseline.

**Khai báo:** code thử thách được Codex viết hỗ trợ. Đây chưa phải bằng chứng tự làm không dùng AI theo yêu cầu thử thách. Học viên cần tự thực hiện lại, giải thích được code và trao đổi với coach về cách xử lý phần đã được AI hỗ trợ; không xóa khai báo để trình bày thành tự làm.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày (~116 tin/s trung bình), SLA p95 < 50 ms, bốn cách đạt độ trễ trong benchmark là hai model BoW và hai embedding head. Embedding + LogReg có F1 SUB cao nhất trong nhóm đạt SLA (0.9744), p95 **37.623 ms**, nhưng throughput một worker chỉ **30.12 tin/s**. Cần đo tải đồng thời, tải đỉnh và tăng năng lực; không suy ra đáp ứng sản lượng chỉ từ p95.

BoW + MLP là baseline triển khai khả thi về tốc độ: p95 **2.408 ms**, throughput cả lô **14284 tin/s**, F1 test **0.9570**. Tuy nhiên F1 khó chỉ 0.488 nên chất lượng ngoài phân phối là đánh đổi lớn. Cần cải thiện dữ liệu và kiểm tra false positive với OTP, giao dịch hợp lệ.

Đề xuất router: model rẻ xử lý tin chắc chắn; các tin có xác suất trong [0.3, 0.7] hoặc model bất đồng được chuyển LLM/hàng đợi xem xét. Hiệu chỉnh ngưỡng trên validation. Nhánh LLM có p95 1052–1661 ms trong lần chạy này nên không đáp ứng SLA 50 ms nếu phải chờ đồng bộ; có thể dùng để đánh giá bổ sung hoặc xem xét bất đồng bộ với chính sách sản phẩm phù hợp.

Theo giả định tính tay của đề: 10 triệu SMS/ngày × 30 ngày, mỗi SMS 100 token input + 1 token output, giá 0.10/0.50 USD mỗi triệu token → **3150 USD/tháng** cho LLM nhỏ. Model lớn theo bảng giá lab đắt gấp 10 lần → **31500 USD/tháng**. Chi phí đo từ prompt zero-shot trong lab ngoại suy là 0.00883 USD/1k × 300000 = **2649 USD/tháng**, khác 3150 vì giả định token khác nhau. Đây đều là ước tính, chưa gồm retry, hạ tầng và tải đỉnh.

## 7. AI log và điều đã học

| Phần | Yêu cầu gửi AI / hỗ trợ đã nhận | Kiểm chứng và giới hạn |
| --- | --- | --- |
| Phần 1–3 | Theo khai báo trong notebook: ChatGPT hỗ trợ code tham khảo cho vectorize, sigmoid, gradient descent, metrics, MLP, cosine, top-k, kNN và giải thích | Output kiểm tra các hàm đạt; học viên cần tự đọc lại và giải thích. Các phần này có yêu cầu tự code; không coi việc chạy test đạt là đã tuân thủ yêu cầu tự làm. |
| Phần 4 | Hỗ trợ hàm API, prompt zero/few-shot, parser và batch classification; sửa docstring và chẩn đoán output invalid | Mock test đạt. Sau tăng ngân sách output từ 3 lên 32 token, các thí nghiệm LLM ghi invalid=0. |
| Cell 41–100 | Yêu cầu Codex đọc notebook, hoàn thành các cell thiếu, chạy thí nghiệm và tổng hợp kết quả | Có khai báo dự đoán do AI hỗ trợ; notebook và JSON hiện ghi 15/15 kiểm tra, 10 dự đoán. Không gán dự đoán AI thành dự đoán tự nghĩ của học viên. |
| Thử thách vocab | Codex viết thí nghiệm vocab 100 và 5000 | Có bảng kết quả; học viên chưa chứng minh tự làm phần thử thách không dùng AI. |
| Báo cáo Markdown | Yêu cầu sửa báo cáo theo repo và số liệu hiện tại | AI đọc lại notebook/JSON, dùng đúng p95 và Pareto lần chạy mới. Không chạy lại API để tạo số liệu mới trong lần sửa báo cáo này. |
| Ảnh tính toán | Yêu cầu sinh lại ảnh đúng đề, giữ tên hand_calc.jpg | AI tạo infographic tham khảo có nhãn nguồn AI. Cần thay bằng ảnh tự viết tay trước khi nộp. |

**Gợi ý phản ánh để học viên tự review:** Kết quả cho thấy model lớn và few-shot không tự động nâng F1; cần so chất lượng cùng độ trễ, chi phí và độ bền. Embedding cũng không luôn thắng BoW ở từng ví dụ, và điểm cao trên test cùng phân phối có thể che giấu lỗi ở spam giọng tự nhiên. Những nhận xét này được AI tổng hợp từ số liệu, không phải xác nhận trải nghiệm cá nhân của học viên.

## Các việc còn lại trước khi nộp

- Tự làm lại phần thử thách theo yêu cầu và giữ AI log trung thực.
- Tự viết 5 phép tính trên giấy, chụp rõ các bước, thay hand_calc.jpg; bản AI hiện tại chỉ là tham khảo.
- Bổ sung câu hỏi trong notebook: cell 62, 78, 84, 91; kết quả thử thách ở cell 99. Sửa p95 cũ 36.741 thành 37.623 ms khi mô tả lần chạy hiện tại; Pareto phải có MLP head.
- Tạo repo cá nhân Public đúng mẫu, commit các file bài làm và push; remote của clone lúc kiểm tra vẫn là repo mẫu của lớp.
