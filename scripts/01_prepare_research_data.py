from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "dataset_vietscam_500_original.csv"
SEED = 20261002


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_generator_templates():
    # The original generator is not required at runtime; these exact templates are
    # copied from the frozen source solely to recover scenario/template identity.
    scam_templates = [
        ("Giả danh cơ quan nhà nước", "{role} thong bao: SDT cua ban dang lien quan den duong day rua tien. Vui long click vao link {link} de khai bao tai san gap."),
        ("Giả danh cơ quan nhà nước", "Dạ em chào anh/chị, em gọi từ {role}. Hiện anh chị đang có biên lai phạt nguội vi phạm giao thông chưa nộp. Yêu cầu chuyển {amount} vào STK tạm giữ để xử lý."),
        ("Giả danh cơ quan nhà nước", "Tai khoan VNeID cua ban chua duoc dinh danh. Truy cap {link} de cap nhat ngay keo bi khoa vinh vien."),
        ("Giả danh cơ quan nhà nước", "Đây là cục viễn thông, số điện thoại của bạn sẽ bị khóa sau 2 giờ do phát tán tin nhắn rác. Bấm phím 1 để gặp tổng đài viên."),
        ("Việc nhẹ lương cao", "Tuyển CTV xem video tiktok thả tim kiếm {amount}/ngày. K cọc k vốn. Lh zalo 098xxx để nhận việc."),
        ("Việc nhẹ lương cao", "Cty Shopee thông báo bạn đã trúng tuyển. Cần nạp {amount} để kích hoạt tài khoản làm nhiệm vụ chốt đơn hoàn tiền gốc + 20% lãi."),
        ("Việc nhẹ lương cao", "CTV đánh giá sản phẩm trên Google Maps, mỗi lượt nhận {amount}. Nạp phí duy trì nhóm 300k để bắt đầu làm ngay."),
        ("Việc nhẹ lương cao", "Chị ơi nạp nốt đơn {amount} này là xong nhiệm vụ, cty sẽ hoàn trả toàn bộ gốc và hoa hồng vào thẻ {bank} cho chị luôn."),
        ("Giả mạo ngân hàng / Phishing", "{bank} thong bao: Tai khoan cua ban phat sinh giao dich {amount}. Neu khong phai ban, truy cap {link} de huy."),
        ("Giả mạo ngân hàng / Phishing", "The tin dung {bank} cua ban bi tru phi thuong nien {amount}. Truy cap link {link} nhap ma OTP de duoc hoan tien."),
        ("Giả mạo ngân hàng / Phishing", "Tài khoản iCloud của bạn bị khóa bảo mật. Đăng nhập {link} để khôi phục."),
        ("Giả mạo ngân hàng / Phishing", "Ban nhan dc {amount} tu {name}. Click {link} de xac nhan nhan tien."),
        ("Hack tài khoản / Deepfake mượn tiền", "{friend} ơi đag rảnh k chuyển tao mượn tạm {amount} tí chiều t lấy lương t trả luôn. App {bank} tao đag lỗi k ck dc."),
        ("Hack tài khoản / Deepfake mượn tiền", "Con dang o benh vien can dong gap {amount} tien vien phi. {relative} chuyen vao stk cua bac si {name} ({bank}) giup con voi."),
        ("Hack tài khoản / Deepfake mượn tiền", "(Video call mờ nhòe) {friend} à tao đag kẹt mạng yếu quá, mày bắn qua stk này cho tao {amount} trc nhé."),
        ("Hack tài khoản / Deepfake mượn tiền", "Ck e {amount} vao the {bank} nay nhe, e dang dung mua do ma quen vi o nha. Nhanh nha a."),
        ("Trúng thưởng / Tặng quà", "Chuc mung sdt cua ban trung thuong 1 xe SH va {amount}. Truy cap {link} dong phi ho so 500k de nhan giai."),
        ("Trúng thưởng / Tặng quà", "Bạn là KH thứ 1000 may mắn nhận lọ nước hoa trị giá {amount}. Vui lòng thanh toán tiền thuế hải quan 150k cho shipper."),
        ("Trúng thưởng / Tặng quà", "ZaloPay tang ban the qua tang {amount}. Dang nhap {link} de rinh qua vao vi."),
        ("Tống tiền / Bạo lực mạng", "Tao đang giữ clip của mày. Chuyển {amount} vào {bank} nếu ko tối nay tao up hết lên facebook trường mày."),
        ("Tống tiền / Bạo lực mạng", "Cty casting model. Em gửi 3 tấm ảnh mặc đồ lót qua đây để a xem body duyệt hồ sơ nhé."),
    ]
    safe_templates = [
        ("Giao tiếp gia đình an toàn", "{relative} ơi tối nay nhà mình ăn gì thế? Con tan làm trễ chắc 7h mới về tới nhà."),
        ("Giao tiếp gia đình an toàn", "Sáng nay đi chợ mua cho {relative} 1 kg thịt bò với mớ rau muống nhé. Tiền để trên bàn phòng khách ấy."),
        ("Giao tiếp gia đình an toàn", "Con đóng tiền học tiếng Anh {amount} chưa? Nếu chưa thì bảo để {relative} chuyển khoản cho thầy nhé."),
        ("Giao tiếp gia đình an toàn", "Cuối tuần này nghỉ lễ, cả nhà mình qua nội chơi không? Bố bảo mua ít trái cây mang sang."),
        ("Giao tiếp bạn bè an toàn", "Ê {friend} nay rảnh không, tối 8h ra quán trà chanh cũ chém gió tí không, tao mời."),
        ("Giao tiếp bạn bè an toàn", "Hôm qua đi ăn tổng hết {amount} nhé, tao chia đều rồi mày bắn qua {bank} cho tao đi."),
        ("Giao tiếp bạn bè an toàn", "Trời mưa to quá t đag trú mưa. Tí mày qua đón tao ở cổng trường nha {friend}."),
        ("Giao tiếp bạn bè an toàn", "Happy birthday {friend} nha! Chúc mày tuổi mới mau giàu, bớt hâm lại và sớm có người yêu."),
        ("Giao tiếp công việc an toàn", "Sếp ơi, file báo cáo tháng này em gửi qua email rồi ạ. Sếp xem có cần chỉnh sửa gì không báo em nhé."),
        ("Giao tiếp công việc an toàn", "Chiều nay 3h toàn team mình họp chốt KPIs tuần nhé. Mọi người mang theo laptop."),
        ("Giao tiếp công việc an toàn", "Dạ em cảm ơn anh chị đã phỏng vấn em ạ, em sẽ chờ kết quả từ phòng nhân sự ạ."),
        ("Giao tiếp công việc an toàn", "Chị nhân sự báo tháng này em được thưởng KPI {amount}, check lại tài khoản {bank} xem tiền về chưa em."),
        ("Giao tiếp mua bán/học tập an toàn", "Shop ơi đơn hàng áo thun size M màu đen của em bao giờ thì giao ạ? Em đợi từ qua tới giờ."),
        ("Giao tiếp mua bán/học tập an toàn", "Dạ em nhận được hàng rồi nha shop, vừa in luôn, tí em chuyển {amount} tiền hàng qua {bank} cho shop nhé."),
        ("Giao tiếp mua bán/học tập an toàn", "Thầy ơi bài tập lớn nhóm 3 nộp lúc 11h tối qua trên hệ thống rồi ạ, thầy check giúp nhóm em nha."),
        ("Giao tiếp mua bán/học tập an toàn", "Lịch thi môn Toán dời sang thứ 3 tuần sau nhé các bạn. Chú ý ôn tập đầy đủ."),
    ]
    return scam_templates, safe_templates


def template_regex(template: str):
    parts = re.split(r"(\{[^{}]+\})", template)
    expr = ""
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            expr += r".+?"
        else:
            expr += re.escape(p)
    return re.compile("^" + expr + "$", re.IGNORECASE)


def assign_scenario_ids(df: pd.DataFrame) -> pd.DataFrame:
    scam, benign = load_generator_templates()
    patterns = []
    for label, templates, prefix in [(1, scam, "SCAM"), (0, benign, "BENIGN")]:
        for idx, (category, tmpl) in enumerate(templates):
            patterns.append((label, idx, category, tmpl, template_regex(tmpl), f"{prefix}_{idx:02d}"))

    scenario_ids, template_texts = [], []
    for _, row in df.iterrows():
        text = row["text"]
        matches = [p for p in patterns if p[0] == int(row["label"]) and p[4].match(text)]
        if len(matches) != 1:
            raise RuntimeError(f"Template matching failed: {len(matches)} matches for {text!r}")
        _, _, _, tmpl, _, sid = matches[0]
        scenario_ids.append(sid)
        template_texts.append(tmpl)

    out = df.copy()
    out["scenario_id"] = scenario_ids
    out["source_template"] = template_texts
    return out


def prepare_development():
    raw = pd.read_csv(SOURCE)
    raw["text"] = raw["text"].astype(str).str.strip()
    raw["label"] = raw["label"].astype(int)
    raw = raw[raw["text"].ne("") & raw["label"].isin([0, 1])].copy()

    # Remove any contradictory text before exact deduplication.
    nlabels = raw.groupby("text")["label"].nunique()
    conflicts = set(nlabels[nlabels > 1].index)
    if conflicts:
        raw = raw[~raw["text"].isin(conflicts)]

    unique = raw.drop_duplicates("text", keep="first").reset_index(drop=True)
    unique = assign_scenario_ids(unique)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    unique["fold"] = -1
    X = unique["text"].values
    y = unique["label"].values
    groups = unique["scenario_id"].values

    for fold, (_, test_idx) in enumerate(sgkf.split(X, y, groups)):
        unique.loc[test_idx, "fold"] = fold

    # Whole scenario groups remain isolated. Fold 0=validation, fold 1=scenario diagnostic test.
    unique["split"] = "train"
    unique.loc[unique["fold"] == 0, "split"] = "validation"
    unique.loc[unique["fold"] == 1, "split"] = "scenario_test"

    unique.to_csv(DATA / "development_unique_with_scenario.csv", index=False, encoding="utf-8-sig")
    for split, filename in [
        ("train", "dev_train.csv"),
        ("validation", "dev_validation.csv"),
        ("scenario_test", "dev_scenario_test.csv"),
    ]:
        unique[unique["split"] == split].to_csv(DATA / filename, index=False, encoding="utf-8-sig")

    return {
        "raw_rows": int(len(pd.read_csv(SOURCE))),
        "unique_rows": int(len(unique)),
        "exact_duplicates_removed": int(len(pd.read_csv(SOURCE)) - len(unique)),
        "conflicting_texts_removed": int(len(conflicts)),
        "split_counts": unique["split"].value_counts().to_dict(),
        "split_label_counts": {
            s: unique[unique["split"] == s]["label"].value_counts().sort_index().to_dict()
            for s in ["train", "validation", "scenario_test"]
        },
        "scenario_counts": {f"{k[0]}|label={k[1]}": int(v) for k, v in unique.groupby(["split", "label"])["scenario_id"].nunique().to_dict().items()},
    }


BANKS = ["Vietcombank", "BIDV", "MBBank", "ACB", "TPBank", "VIB", "Techcombank", "Sacombank"]
AMOUNTS = ["450 nghìn", "1,2 triệu", "2 triệu", "3,5 triệu", "5 triệu", "8 triệu", "12 triệu", "25 triệu"]
LINKS = ["xacminh-taikhoan247.com", "kiemtra-giaodich.net", "hotro-vneid.info", "quatang-khachhang.vip", "hoantien-donhang.site"]
NAMES = ["Minh", "Huy", "Trang", "Lan", "Phúc", "Nam", "Thảo", "An"]


def expand_templates(scenario_id, label, category, templates, n=10, seed_offset=0):
    rng = random.Random(SEED + seed_offset)
    rows = []
    seen = set()
    attempts = 0
    while len(rows) < n and attempts < 1000:
        attempts += 1
        tmpl = rng.choice(templates)
        values = {
            "bank": rng.choice(BANKS),
            "amount": rng.choice(AMOUNTS),
            "link": rng.choice(LINKS),
            "name": rng.choice(NAMES),
            "code": str(rng.randint(100000, 999999)),
            "hour": rng.choice(["30 phút", "1 giờ", "2 giờ", "trước 17h hôm nay"]),
        }
        text = tmpl.format(**values)
        if text in seen:
            continue
        seen.add(text)
        rows.append({
            "text": text,
            "label": label,
            "scenario_id": scenario_id,
            "category": category,
            "source_type": "author_curated_synthetic_post_freeze",
            "variant_id": f"{scenario_id}_V{len(rows)+1:02d}",
        })
    if len(rows) != n:
        raise RuntimeError(f"Could not generate {n} unique samples for {scenario_id}")
    return rows


def create_independent_test():
    scam = {
        "IND_S01_AUTHORITY": ("Mạo danh cơ quan", [
            "Tôi phụ trách hồ sơ của anh tại cơ quan điều tra. Hồ sơ đang có vấn đề, cần chuyển {amount} vào tài khoản bảo chứng trước {hour} để đối chiếu.",
            "Thông báo khẩn: căn cước của bạn đang bị sử dụng trong một vụ án. Mở {link} và hoàn tất xác minh tài sản ngay hôm nay.",
            "Bên thanh tra thông báo bạn còn khoản phạt chưa xử lý. Hãy nộp {amount} vào tài khoản tạm giữ, sau khi xác minh sẽ hoàn lại.",
            "Tôi là cán bộ hỗ trợ định danh. Tài khoản VNeID của anh có nguy cơ bị khóa; truy cập {link} để cập nhật thông tin cá nhân.",
            "Cơ quan chức năng yêu cầu giữ bí mật cuộc gọi và chuyển {amount} để chứng minh nguồn tiền hợp pháp trong vòng {hour}.",
        ]),
        "IND_S02_BANK_PHISH": ("Giả mạo ngân hàng/phishing", [
            "{bank}: hệ thống vừa ghi nhận giao dịch {amount}. Nếu bạn không thực hiện, xác nhận hủy tại {link} ngay.",
            "Thẻ {bank} của quý khách đang chờ hoàn phí. Đăng nhập {link} và nhập mã OTP để nhận lại tiền.",
            "Tài khoản {bank} cần xác minh thiết bị mới. Vui lòng mở {link} rồi điền thông tin đăng nhập trước {hour}.",
            "Bạn vừa nhận {amount}. Để tiền được cộng vào số dư {bank}, hãy xác nhận tại {link}.",
            "Cảnh báo từ {bank}: tài khoản có dấu hiệu bất thường. Bấm {link} để mở khóa dịch vụ ngân hàng số.",
        ]),
        "IND_S03_TASK": ("Nhiệm vụ/hoa hồng", [
            "Công việc online tại nhà: mỗi đơn nhận 15% hoa hồng. Bạn nạp {amount} vào ví nhiệm vụ để bắt đầu đơn đầu tiên.",
            "Nhóm cộng tác viên đang thiếu một suất. Hoàn thành đơn ảo rồi hệ thống trả cả vốn lẫn hoa hồng; cần ứng trước {amount}.",
            "Chỉ cần đánh giá 5 sao sản phẩm là có thu nhập. Muốn mở cấp VIP rút tiền, nạp {amount} trước {hour}.",
            "Bạn đã hoàn thành 2 nhiệm vụ. Nạp thêm {amount} để ghép đơn cuối và rút toàn bộ tiền thưởng.",
            "Tuyển người thả tim video, không cần kinh nghiệm. Kích hoạt tài khoản bằng khoản đảm bảo {amount}, sau đó hoàn ngay.",
        ]),
        "IND_S04_ACCOUNT_TAKEOVER": ("Mượn tiền giả mạo", [
            "{name} đây, máy mình hỏng nên đang nhắn nhờ máy người khác. Chuyển giúp mình {amount} vào tài khoản mới, lát mình trả.",
            "T đang ở bệnh viện không gọi được. Bắn gấp {amount} vào số tài khoản t gửi, đừng gọi lại vì bác sĩ đang làm thủ tục.",
            "Anh ơi em đổi tài khoản ngân hàng rồi, chuyển giúp em {amount} vào số mới này với, đang cần thanh toán gấp.",
            "Mạng yếu nên video cứ đứng hình. Mày chuyển trước {amount} cho tao vào tài khoản này nhé, tối tao gọi lại.",
            "Con đang có việc khẩn, tài khoản cũ bị lỗi. Mẹ chuyển hộ {amount} sang tài khoản của bạn con trước {hour} nhé.",
        ]),
        "IND_S05_PRIZE": ("Trúng thưởng/phí nhận quà", [
            "Chúc mừng bạn được chọn nhận quà trị giá {amount}. Để giao quà, vui lòng đóng 190 nghìn phí hồ sơ trước {hour}.",
            "Số điện thoại của bạn trúng giải khách hàng may mắn. Nhận thưởng tại {link} và thanh toán phí xác nhận 250 nghìn.",
            "Bạn vừa nhận voucher đặc biệt từ chương trình tri ân. Nộp phí vận chuyển 180 nghìn để mở khóa phần thưởng {amount}.",
            "Giải thưởng của bạn đang chờ xác minh. Chuyển 300 nghìn phí thuế, hệ thống sẽ giải ngân {amount} ngay sau đó.",
            "Tài khoản được chọn trúng quà ngẫu nhiên. Đăng nhập {link} và đóng phí phát hành để nhận trong hôm nay.",
        ]),
        "IND_S06_EXTORTION": ("Tống tiền trực tuyến", [
            "Tao có ảnh riêng tư của mày. Chuyển {amount} trước {hour}, nếu không tao gửi cho toàn bộ danh bạ.",
            "Video cuộc gọi tối qua đã được ghi lại. Muốn xóa thì nộp {amount}; chậm là tao đăng công khai.",
            "Tài khoản của mày đã bị lấy dữ liệu. Gửi {amount} vào ví này nếu không toàn bộ ảnh sẽ được phát tán.",
            "Đừng báo ai. Tao đang giữ đoạn clip của mày và danh sách bạn bè. Chuyển {amount} để tao xóa ngay.",
            "Nếu không thanh toán {amount} trong {hour}, hình ảnh nhạy cảm sẽ được gửi cho gia đình và đồng nghiệp.",
        ]),
        "IND_S07_FAKE_REFUND": ("Giả mạo hỗ trợ mua sắm/hoàn tiền", [
            "Bộ phận chăm sóc khách hàng báo đơn của bạn bị lỗi. Muốn hoàn {amount}, cài ứng dụng hỗ trợ từ {link} rồi đăng nhập ngân hàng.",
            "Sản phẩm bạn vừa nhận thuộc lô cần thu hồi. Truy cập {link} để khai số tài khoản và nhận khoản hoàn {amount}.",
            "Shop cần hoàn tiền cho đơn cũ. Bạn cung cấp mã OTP vừa gửi để hệ thống xác nhận hoàn {amount}.",
            "Đơn hàng đang treo hoàn phí. Nhân viên yêu cầu bấm {link} và bật chia sẻ màn hình để xử lý ngay.",
            "Bên vận chuyển gọi báo hoàn tiền vì giao sai. Hãy đăng nhập ngân hàng theo đường dẫn {link} để nhận {amount}.",
        ]),
        "IND_S08_RECRUITMENT": ("Tuyển dụng thu phí", [
            "Hồ sơ của bạn đã đạt vòng đầu. Để giữ suất làm việc, chuyển {amount} tiền đồng phục và đào tạo trước {hour}.",
            "Công ty tuyển nhân viên nhập liệu tại nhà, lương cao. Ứng viên cần đóng cọc thiết bị {amount} trước khi nhận việc.",
            "Bạn đã được nhận vào vị trí online. Phí mở tài khoản nhân sự là {amount}, hoàn lại sau tháng đầu tiên.",
            "Phòng tuyển dụng yêu cầu nộp {amount} phí hồ sơ để cấp mã nhân viên và lịch nhận việc.",
            "Cơ hội việc làm không cần phỏng vấn. Chỉ cần chuyển {amount} phí bảo đảm, sau đó công ty gửi máy làm việc.",
        ]),
    }

    benign = {
        "IND_B01_WORK": ("Công việc hợp lệ", [
            "Chiều nay nhóm mình họp lúc 14h để rà lại kế hoạch tháng, mọi người xem file em vừa gửi trong hệ thống nội bộ nhé.",
            "Em đã cập nhật báo cáo tuần lên thư mục chung, anh {name} xem giúp phần ngân sách có cần sửa không.",
            "Phòng nhân sự xác nhận lịch phỏng vấn của bạn vào sáng thứ Hai tại văn phòng; không cần nộp bất kỳ khoản phí nào.",
            "Chị gửi bảng chấm công tháng này rồi, mọi người kiểm tra số ngày phép và phản hồi trước cuối ngày.",
            "Buổi đào tạo nội bộ chuyển sang phòng 305. Nhớ mang laptop và đăng nhập tài khoản công ty trước khi bắt đầu.",
        ]),
        "IND_B02_FAMILY": ("Gia đình hợp lệ", [
            "Mẹ đã chuyển {amount} tiền học vào tài khoản của con như mình nói tối qua, con kiểm tra giúp mẹ nhé.",
            "Chiều con ghé siêu thị mua đồ giúp ba, tiền mặt ba để trong ngăn kéo bàn ăn rồi.",
            "Anh về trễ thì nhắn em biết, em để phần cơm trong tủ lạnh nhé.",
            "Con đóng viện phí cho bà rồi thì gửi biên lai vào nhóm gia đình để mọi người tiện theo dõi nha.",
            "Cuối tuần cả nhà đi ăn, chị đặt bàn trước rồi; ai bận thì báo sớm để chị đổi số người.",
        ]),
        "IND_B03_BANK_LEGIT": ("Thông báo tài chính hợp lệ", [
            "Tôi vừa tự chuyển {amount} từ tài khoản {bank} sang tài khoản tiết kiệm và giao dịch đã hiển thị thành công.",
            "Ứng dụng {bank} thông báo giao dịch mua hàng {amount} đã hoàn tất; tôi kiểm tra đúng là giao dịch của mình.",
            "Mình đã khóa thẻ trực tiếp trong ứng dụng {bank}; mai sẽ ra chi nhánh làm lại thẻ mới.",
            "Ngân hàng gửi sao kê tháng qua ứng dụng chính thức, mình đã tải về để đối chiếu chi tiêu.",
            "Khoản hoàn tiền {amount} đã xuất hiện trong lịch sử giao dịch của {bank}; không có yêu cầu nhập OTP hay đăng nhập link lạ.",
        ]),
        "IND_B04_ECOMMERCE": ("Mua bán hợp lệ", [
            "Shop ơi mình nhận được hàng rồi, sản phẩm đúng mẫu. Tối nay mình đánh giá đơn trên ứng dụng nhé.",
            "Đơn của mình đang hiển thị giao ngày mai, cho hỏi shipper có gọi trước khi đến không ạ?",
            "Mình đã thanh toán {amount} trực tiếp trên sàn, shop kiểm tra trạng thái đơn giúp mình nhé.",
            "Sản phẩm bị sai màu nên mình vừa tạo yêu cầu đổi trả trong ứng dụng chính thức, mã yêu cầu là {code}.",
            "Mình muốn mua thêm một sản phẩm giống đơn cũ, shop còn hàng không để mình đặt trực tiếp trên sàn?",
        ]),
        "IND_B05_EDUCATION": ("Học tập hợp lệ", [
            "Thầy cho em hỏi bài tập tuần này nộp trên LMS trước 23h59 Chủ nhật đúng không ạ?",
            "Em đã đóng học phí {amount} tại cổng thanh toán của trường và lưu biên lai trong tài khoản sinh viên.",
            "Lớp mình đổi phòng thực hành sang B203, các bạn kiểm tra thông báo trên cổng đào tạo nhé.",
            "Nhóm em gửi bản nháp báo cáo qua email trường rồi, cô xem giúp phần tài liệu tham khảo ạ.",
            "Nhà trường thông báo lịch thi mới trên website chính thức; mình đã đăng nhập cổng sinh viên để kiểm tra.",
        ]),
        "IND_B06_FRIENDS": ("Bạn bè hợp lệ", [
            "Hôm qua nhóm mình ăn hết {amount}, tao đã chia phần trong app nhóm rồi, lúc nào rảnh mày chuyển cũng được.",
            "Tối nay đá bóng 7h sân cũ nha, ai đến trễ thì nhắn trong nhóm để tụi tao chờ.",
            "Mày gửi lại cho tao ảnh chuyến đi hôm trước với, máy tao đổi nên bị mất mấy tấm.",
            "Tao đặt vé xem phim rồi, tiền vé mỗi người 120 nghìn, gặp nhau ở rạp rồi tính cũng được.",
            "Mai tao qua trả cuốn sách mượn tuần trước, chiều mày có ở nhà không?",
        ]),
        "IND_B07_PAYROLL": ("Lương/thưởng hợp lệ", [
            "Phòng nhân sự thông báo thưởng quý {amount}; nhân viên chỉ cần kiểm tra phiếu lương trên cổng nội bộ, không cung cấp OTP.",
            "Lương tháng này đã được chuyển qua {bank}. Nếu số tiền chưa đúng, gửi phản hồi bằng email công ty cho kế toán.",
            "Kế toán vừa cập nhật phụ cấp công tác vào bảng lương, mọi người kiểm tra trên hệ thống nhân sự nhé.",
            "Khoản hoàn ứng {amount} đã được duyệt, dự kiến về tài khoản trong hai ngày làm việc tới.",
            "Công ty không yêu cầu nhân viên đóng bất kỳ khoản phí nào để nhận thưởng cuối năm; thông tin chi tiết ở portal nội bộ.",
        ]),
        "IND_B08_TRAVEL": ("Du lịch/đặt chỗ hợp lệ", [
            "Khách sạn đã xác nhận đặt phòng của mình trong ứng dụng chính thức, mã đặt chỗ {code} và thanh toán tại quầy.",
            "Mình vừa đổi ngày bay trực tiếp trên website hãng, email xác nhận mới đã gửi về hộp thư.",
            "Tour cuối tuần còn hai chỗ, mình sẽ đến văn phòng công ty lữ hành ký hợp đồng trước khi thanh toán.",
            "Nhà nghỉ cho phép hủy miễn phí trước thứ Sáu; mình kiểm tra điều kiện ngay trong ứng dụng đặt phòng.",
            "Mình đã mua vé tàu trên ứng dụng chính thức, QR vé đang lưu trong tài khoản của mình.",
        ]),
    }

    rows = []
    off = 0
    for sid, (cat, tmpls) in scam.items():
        rows.extend(expand_templates(sid, 1, cat, tmpls, n=10, seed_offset=off)); off += 17
    for sid, (cat, tmpls) in benign.items():
        rows.extend(expand_templates(sid, 0, cat, tmpls, n=10, seed_offset=off)); off += 17

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "sample_id", [f"IND_{i+1:03d}" for i in range(len(df))])

    # Guard against exact overlap with development corpus.
    dev = pd.read_csv(DATA / "development_unique_with_scenario.csv")
    overlap = set(df["text"]).intersection(set(dev["text"]))
    if overlap:
        raise RuntimeError(f"Independent set exact-overlaps development: {len(overlap)}")

    df.to_csv(DATA / "independent_test_v1.csv", index=False, encoding="utf-8-sig")
    return {
        "rows": len(df),
        "labels": df["label"].value_counts().sort_index().to_dict(),
        "scenarios": int(df["scenario_id"].nunique()),
        "exact_overlap_with_development": 0,
        "source_type": "author_curated_synthetic_post_freeze",
    }


def create_ood_gray():
    rng = random.Random(SEED + 999)
    rows = []

    scam_scenarios = {
        "OOD_S01_ROMANCE_INVEST": [
            "Anh nói chuyện với em mấy tháng rồi nên tin em nhé. Nạp {amount} vào sàn này để cùng đầu tư, em sẽ chỉ cách rút lãi.",
            "Người yêu quen qua mạng bảo có kèo đầu tư nội bộ, yêu cầu chuyển {amount} sang ví lạ để mở tài khoản VIP.",
            "Em đang ở nước ngoài nhưng muốn xây tương lai với anh. Trước tiên anh giúp em nạp {amount} vào nền tảng đầu tư này.",
            "Bạn trai online gửi ảnh lợi nhuận và bảo mình vay tiền nạp {amount} vào sàn để không bỏ lỡ cơ hội.",
            "Sau vài tuần làm quen, người đó rủ mua tiền số qua link riêng và yêu cầu nạp thử {amount}.",
        ],
        "OOD_S02_VISA": [
            "Dịch vụ cam kết visa 5 năm không cần phỏng vấn, chỉ cần chuyển cọc {amount} hôm nay để giữ suất.",
            "Bên môi giới nói có thể làm đẹp hộ chiếu và bảo chuyển {amount} vào tài khoản cá nhân trước khi ký hợp đồng.",
            "Muốn chắc chắn đậu visa, họ yêu cầu nộp thêm {amount} phí quan hệ và không xuất hóa đơn.",
            "Người tự xưng nhân viên lãnh sự hứa cấp visa nhanh nếu chuyển {amount} vào tài khoản riêng.",
            "Đơn vị quảng cáo đi lao động nước ngoài miễn phỏng vấn, yêu cầu đóng {amount} để mở hồ sơ qua Zalo.",
        ],
        "OOD_S03_TECH_SUPPORT": [
            "Nhân viên hỗ trợ nói máy bạn nhiễm virus, yêu cầu cài phần mềm điều khiển từ xa và đăng nhập ngân hàng để kiểm tra.",
            "Cuộc gọi tự xưng kỹ thuật viên đề nghị bật chia sẻ màn hình rồi đọc mã OTP để xử lý lỗi tài khoản.",
            "Thông báo giả cho biết máy tính bị khóa bản quyền; muốn mở phải chuyển {amount} và cài công cụ từ link lạ.",
            "Người lạ nói tài khoản email bị hack và yêu cầu cấp quyền remote desktop để sửa ngay.",
            "Hỗ trợ viên không rõ nguồn gốc bảo tải ứng dụng ngoài cửa hàng và nhập mật khẩu ngân hàng để hoàn phí dịch vụ.",
        ],
        "OOD_S04_CHARITY": [
            "Tài khoản lạ kêu gọi cứu trợ khẩn cấp, thúc giục chuyển {amount} vào tài khoản cá nhân nhưng không có thông tin tổ chức xác minh.",
            "Fanpage mới lập đăng ảnh bệnh nhi và yêu cầu chuyển tiền ngay vào số tài khoản cá nhân, không công bố hồ sơ hay đơn vị tiếp nhận.",
            "Tin nhắn nói quỹ từ thiện cần đóng góp trong 30 phút để được nhân đôi tiền, kèm số tài khoản riêng.",
            "Người tự nhận đại diện quỹ đề nghị nộp phí {amount} để được cấp chứng nhận nhà tài trợ rồi mới nhận biên lai.",
            "Bài đăng dùng ảnh thiên tai cũ, yêu cầu chuyển tiền gấp và chặn mọi câu hỏi về nguồn xác minh.",
        ],
        "OOD_S05_RECOVERY": [
            "Người tự xưng luật sư nói có thể lấy lại tiền đã bị lừa, nhưng yêu cầu đóng trước {amount} phí truy hồi.",
            "Dịch vụ hỗ trợ nạn nhân hứa hoàn 100% tiền mất nếu chuyển {amount} để mở hồ sơ điều tra.",
            "Tài khoản Telegram bảo đã tìm thấy ví của kẻ lừa đảo và cần nộp {amount} phí gas để thu hồi tiền.",
            "Người lạ liên hệ sau vụ lừa cũ, nói có quan hệ với ngân hàng và yêu cầu phí {amount} trước khi giải ngân lại.",
            "Trang web cam kết truy vết tiền số bị mất, bắt nhập seed phrase và đóng {amount} phí kích hoạt.",
        ],
        "OOD_S06_RENTAL": [
            "Chủ nhà chỉ gửi ảnh căn hộ, từ chối xem trực tiếp và yêu cầu cọc {amount} trong hôm nay để giữ phòng.",
            "Tin đăng thuê nhà giá rẻ yêu cầu chuyển {amount} trước khi cung cấp địa chỉ cụ thể vì đang ở nước ngoài.",
            "Người cho thuê thúc giục cọc ngay vào tài khoản người khác, không ký hợp đồng và không cho xem giấy tờ.",
            "Căn hộ được quảng cáo thấp hơn thị trường, chủ nhà nói có nhiều người hỏi và phải chuyển {amount} mới được xem nhà.",
            "Môi giới giả yêu cầu phí giữ chỗ {amount} rồi hẹn sau mới gửi hợp đồng và thông tin căn hộ.",
        ],
    }

    benign_scenarios = {
        "OOD_B01_INVEST_DISCUSS": [
            "Mình đang đọc báo cáo tài chính công ty, chưa quyết định đầu tư và sẽ chỉ giao dịch qua tài khoản chứng khoán chính chủ.",
            "Anh em tối nay bàn thử về quỹ ETF nhé, không ai cần chuyển tiền cho ai cả.",
            "Tôi đã đặt lệnh nhỏ trên ứng dụng của công ty chứng khoán mà mình đăng ký từ trước.",
            "Nhóm học tài chính đang phân tích rủi ro tiền số, mục tiêu là học chứ không kêu gọi góp vốn.",
            "Tôi nhận được tư vấn đầu tư nhưng sẽ đến chi nhánh xác minh trước khi thực hiện bất kỳ giao dịch nào.",
        ],
        "OOD_B02_VISA_LEGIT": [
            "Tôi đã nộp hồ sơ visa trực tiếp trên cổng chính thức và lịch hẹn sinh trắc học đã được xác nhận.",
            "Công ty du học gửi hợp đồng dịch vụ và hóa đơn; gia đình sẽ đến văn phòng kiểm tra trước khi thanh toán.",
            "Đại sứ quán thông báo lịch hẹn trong tài khoản hồ sơ của tôi, không yêu cầu chuyển tiền qua cá nhân.",
            "Tôi đang chuẩn bị giấy tờ visa theo danh sách trên website chính thức của cơ quan lãnh sự.",
            "Phí visa được thanh toán tại kênh được công bố trong hồ sơ, mình đã lưu biên lai điện tử.",
        ],
        "OOD_B03_TECH_LEGIT": [
            "IT công ty sẽ qua tận máy để cài bản vá; họ không yêu cầu tôi đọc mật khẩu hay OTP.",
            "Tôi tự mở trang hỗ trợ chính thức của hãng và tạo ticket số {code} cho lỗi máy tính.",
            "Kỹ thuật viên nội bộ hướng dẫn cập nhật phần mềm qua kho ứng dụng công ty, không dùng link ngoài.",
            "Máy lỗi nên tôi mang trực tiếp đến trung tâm bảo hành có phiếu tiếp nhận.",
            "Bộ phận IT yêu cầu đổi mật khẩu trong cổng nội bộ do chính tôi truy cập, không hỏi mật khẩu hiện tại.",
        ],
        "OOD_B04_CHARITY_LEGIT": [
            "Tôi ủng hộ qua tài khoản công khai trên website chính thức của tổ chức và nhận biên lai điện tử.",
            "Cơ quan mình phát động quyên góp tự nguyện, danh sách và đơn vị tiếp nhận được công bố trong email nội bộ.",
            "Gia đình đóng góp cho quỹ học bổng của trường qua cổng thanh toán chính thức.",
            "Tôi kiểm tra thông tin quỹ từ thiện trên trang chính thức trước khi chuyển khoản.",
            "Nhóm thiện nguyện công khai sao kê và cho phép mọi người xác minh trước khi ủng hộ.",
        ],
        "OOD_B05_RENTAL_LEGIT": [
            "Tôi đã xem căn hộ trực tiếp, kiểm tra giấy tờ và sẽ ký hợp đồng trước khi đặt cọc.",
            "Chủ nhà hẹn mình đến xem phòng chiều mai; chưa ai yêu cầu chuyển tiền trước.",
            "Tiền cọc sẽ được ghi rõ trong hợp đồng thuê và thanh toán sau khi hai bên ký.",
            "Môi giới gửi địa chỉ và lịch xem nhà, phí dịch vụ chỉ thanh toán khi giao dịch hoàn tất.",
            "Tôi đang so sánh ba căn hộ và chưa chuyển cọc cho bên nào.",
        ],
        "OOD_B06_CRYPTO_LEGIT": [
            "Tôi đang học cách hoạt động của blockchain bằng tài khoản thử nghiệm, không nạp tiền thật.",
            "Bạn mình gửi bài nghiên cứu về tiền số để đọc, không có lời mời đầu tư hay đường link nạp tiền.",
            "Tôi tự chuyển một khoản nhỏ giữa hai ví của chính mình để kiểm tra mạng lưới.",
            "Nhóm kỹ thuật đang thử smart contract trên testnet nên token không có giá trị tiền thật.",
            "Mình chỉ thảo luận biến động thị trường, không nhận tiền đầu tư hộ bất kỳ ai.",
        ],
    }

    gray_scenarios = {
        "GRAY_G01_TRANSFER": [
            "Chuyển giúp tao {amount} vào tài khoản này nhé, lát tao nói lý do sau.",
            "Em gửi anh số tài khoản mới, anh chuyển {amount} giúp em trước tối nay nha.",
            "Có việc gấp, bắn cho mình {amount} rồi mình gọi lại sau.",
            "Mẹ chuyển hộ con {amount} vào số này với, con đang bận không nói chuyện được.",
            "Chuyển khoản giúp mình {amount}, lát gặp mình giải thích.",
        ],
        "GRAY_G02_LINK": [
            "Bạn mở link này xác nhận giúp mình nhé: {link}",
            "Hệ thống bảo cần vào {link} để cập nhật thông tin, mình có nên làm không?",
            "Bên kia gửi {link} nói là trang xác nhận, bạn xem giúp mình.",
            "Mình vừa nhận đường dẫn {link}, họ bảo bấm vào mới xử lý được.",
            "Có tin nhắn yêu cầu truy cập {link} nhưng không nói rõ đơn vị nào gửi.",
        ],
        "GRAY_G03_OTP": [
            "Mã OTP {code} vừa gửi về máy mình, bên kia đang hỏi mã này.",
            "Họ nói cần OTP để hoàn tất giao dịch, mình chưa biết có nên đưa không.",
            "Điện thoại vừa nhận mã {code}, nhưng mình không nhớ đã yêu cầu thao tác gì.",
            "Có người đang gọi hỏi mã xác thực vừa gửi về máy mình.",
            "Tin nhắn có OTP {code}, mình chưa rõ nó thuộc dịch vụ nào.",
        ],
        "GRAY_G04_JOB": [
            "Có người mời làm việc online thu nhập cao nhưng chưa gửi hợp đồng hay thông tin công ty.",
            "Họ bảo mình được nhận việc rồi, chi tiết phí và quy trình sẽ gửi sau.",
            "Một tài khoản Zalo mời cộng tác nhập dữ liệu tại nhà, chưa rõ công ty nào.",
            "Mình nhận lời mời làm thêm, họ hẹn tối mới giải thích cách nhận lương.",
            "Tin tuyển dụng nói làm online không cần kinh nghiệm, chưa thấy địa chỉ hay website chính thức.",
        ],
        "GRAY_G05_PRIZE": [
            "Tin nhắn nói mình được nhận quà khách hàng, chưa thấy yêu cầu phí hay đường link.",
            "Một bên báo mình trúng thưởng nhưng bảo sẽ gọi lại hướng dẫn sau.",
            "Mình được thông báo có quà tặng, hiện chưa biết chương trình nào tổ chức.",
            "Có SMS nói tài khoản được chọn nhận quà, không ghi rõ cách nhận.",
            "Bên bán hàng báo có phần quà tri ân nhưng chưa cung cấp thông tin xác minh.",
        ],
        "GRAY_G06_URGENT": [
            "Gọi cho mình ngay khi đọc được tin này, có việc rất gấp.",
            "Mình đang gặp chuyện, lát gửi thông tin rồi nhờ bạn giúp nhé.",
            "Có việc khẩn liên quan tài khoản nhưng mình chưa tiện nói qua tin nhắn.",
            "Đừng bỏ qua tin này, tối nay mình cần nhờ một việc quan trọng.",
            "Mình cần bạn hỗ trợ gấp nhưng chờ mình gửi thêm chi tiết.",
        ],
    }

    def add_group(group, expected, kind):
        nonlocal rows
        for sid, tmpls in group.items():
            for j, tmpl in enumerate(tmpls, 1):
                text = tmpl.format(
                    amount=rng.choice(AMOUNTS), link=rng.choice(LINKS),
                    code=str(rng.randint(100000, 999999)), bank=rng.choice(BANKS),
                    name=rng.choice(NAMES), hour=rng.choice(["30 phút", "1 giờ", "trước 17h"]),
                )
                rows.append({
                    "text": text,
                    "expected_action": expected,
                    "scenario_id": sid,
                    "scenario_type": kind,
                    "source_type": "author_curated_synthetic_post_freeze",
                })

    add_group(scam_scenarios, "SCAM", "unseen_scam")
    add_group(benign_scenarios, "BENIGN", "hard_negative_benign")
    add_group(gray_scenarios, "UNCERTAIN", "gray_area")

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "sample_id", [f"OOD_{i+1:03d}" for i in range(len(df))])
    df.to_csv(DATA / "ood_gray_v1.csv", index=False, encoding="utf-8-sig")
    return {
        "rows": len(df),
        "expected_actions": df["expected_action"].value_counts().to_dict(),
        "scenario_types": df["scenario_type"].value_counts().to_dict(),
        "scenarios": int(df["scenario_id"].nunique()),
    }


def create_rag_assets():
    kb = [
        ("K01_AUTHORITY", "Mạo danh cơ quan và yêu cầu xác minh/chuyển tiền", "Kẻ gian giả danh Công an, Viện kiểm sát, Tòa án hoặc cơ quan quản lý, tạo áp lực thời gian, yêu cầu chuyển tiền vào tài khoản 'tạm giữ', khai báo tài sản, cung cấp thông tin hoặc truy cập đường dẫn xác minh.", "https://www.mps.gov.vn/bai-viet/nang-cao-canh-giac-truoc-25-kich-ban-lua-dao-tren-khong-gian-mang-nam-2026-1788865614"),
        ("K02_BANK_PHISH", "Giả mạo ngân hàng và đánh cắp thông tin", "Tin nhắn giả ngân hàng thường thông báo giao dịch bất thường, khóa tài khoản hoặc hoàn phí rồi dẫn nạn nhân tới trang giả mạo để lấy mật khẩu, OTP hoặc thông tin tài chính.", "https://www.mps.gov.vn/bai-viet/canh-bao-lua-dao-chiem-doat-tai-san-tu-su-co-an-ninh-mang-tai-trung-tam-thong-tin-tin-dung-quoc-gia-cic-1757991790"),
        ("K03_TASK", "Lừa đảo làm nhiệm vụ và hoa hồng", "Đối tượng mời làm cộng tác viên, đánh giá sản phẩm, chốt đơn hoặc làm nhiệm vụ nhận hoa hồng; sau các khoản nhỏ tạo lòng tin, nạn nhân bị yêu cầu nạp tiền ngày càng lớn để 'hoàn tất đơn' hoặc rút tiền.", "https://www.mps.gov.vn/bai-viet/cong-an-tinh-dong-nai-phoi-hop-triet-pha-chuyen-an-lua-dao-tren-khong-gian-mang-quy-mo-dac-biet-lon-1770992938"),
        ("K04_ACCOUNT_TAKEOVER", "Giả mạo người thân/bạn bè mượn tiền", "Kẻ gian chiếm quyền tài khoản hoặc dùng deepfake/deepvoice để giả người quen, viện lý do khẩn cấp, lỗi ngân hàng hoặc video kém chất lượng rồi yêu cầu chuyển tiền sang tài khoản khác.", "https://www.mps.gov.vn/bai-viet/nang-cao-canh-giac-truoc-25-kich-ban-lua-dao-tren-khong-gian-mang-nam-2026-1788865614"),
        ("K05_PRIZE", "Trúng thưởng và phí nhận quà", "Nạn nhân được thông báo trúng thưởng, nhận quà hoặc ưu đãi bất ngờ nhưng phải đóng trước phí hồ sơ, thuế, phí vận chuyển hoặc đăng nhập vào trang không chính thức.", "https://www.mps.gov.vn/bai-viet/nang-cao-canh-giac-truoc-25-kich-ban-lua-dao-tren-khong-gian-mang-nam-2026-1788865614"),
        ("K06_EXTORTION", "Tống tiền trực tuyến và sextortion", "Đối tượng đe dọa phát tán ảnh, video hoặc thông tin riêng tư, yêu cầu nạn nhân chuyển tiền hoặc tiếp tục làm theo chỉ dẫn; một biến thể khác là giả cơ quan chức năng để cô lập và thao túng nạn nhân.", "https://www.mps.gov.vn/bai-viet/canh-bao-toi-pham-lua-dao-bang-thu-doan-tong-tien-truc-tuyen-1764129007"),
        ("K07_ECOM_REFUND", "Giả mạo hỗ trợ mua sắm và hoàn tiền", "Kẻ gian nắm thông tin đơn hàng rồi giả nhân viên sàn, cửa hàng hoặc vận chuyển, viện lý do thu hồi/hoàn tiền để dụ cung cấp dữ liệu, cài ứng dụng, chia sẻ màn hình hoặc thao tác ngân hàng.", "https://www.mps.gov.vn/bai-viet/xuat-hien-thu-doan-lua-dao-moi-khi-mua-hang-online-1790562304"),
        ("K08_TRAVEL", "Lừa đảo đặt phòng và du lịch", "Đối tượng giả mạo cơ sở lưu trú, fanpage hoặc người bán tour, đưa giá hấp dẫn và yêu cầu cọc/chuyển tiền; có thể gửi email giả nền tảng đặt phòng kèm link hoặc tệp độc hại.", "https://www.mps.gov.vn/bai-viet/canh-bao-thu-doan-cua-toi-pham-lua-dao-truc-tuyen-loi-dung-tet-nguyen-dan-binh-ngo-2026-va-cac-dip-le-hoi-dau-nam-1769572336"),
        ("K09_RECRUITMENT", "Tuyển dụng giả và thu phí trước", "Tin tuyển dụng giả hứa việc nhẹ, lương cao hoặc làm tại nhà nhưng yêu cầu ứng viên đóng trước phí hồ sơ, đồng phục, đào tạo, thiết bị hoặc tiền bảo đảm trước khi nhận việc.", "https://www.mps.gov.vn/bai-viet/nang-cao-canh-giac-truoc-25-kich-ban-lua-dao-tren-khong-gian-mang-nam-2026-1788865614"),
        ("K10_TICKET", "Lừa đảo bán vé/sự kiện", "Đối tượng dùng tài khoản hoặc bài đăng giả để bán vé concert, sự kiện hay dịch vụ khan hiếm, yêu cầu đặt cọc hoặc thanh toán trước rồi chặn liên lạc sau khi nhận tiền.", "https://www.mps.gov.vn/bai-viet/nang-cao-canh-giac-truoc-25-kich-ban-lua-dao-tren-khong-gian-mang-nam-2026-1788865614"),
        ("K11_INVESTMENT", "Lừa đảo đầu tư và tiền số", "Đối tượng tạo nền tảng hoặc lời mời đầu tư giả, khoe lợi nhuận cao, cho rút khoản nhỏ để tạo lòng tin rồi thúc giục nạp thêm; có thể kết hợp quan hệ tình cảm hoặc người môi giới giả.", "https://www.mps.gov.vn/bai-viet/thong-bao-tim-nguoi-lien-quan-phuc-vu-cong-tac-kiem-tra-xac-minh-nguon-tin-toi-pham-1778744681"),
        ("K12_VISA", "Lừa đảo visa và dịch vụ xuất nhập cảnh", "Dịch vụ giả cam kết visa, việc làm hoặc 'làm đẹp hộ chiếu', yêu cầu chuyển cọc/phí vào tài khoản cá nhân, thiếu hợp đồng hoặc giấy tờ xác minh và sau đó cắt liên lạc.", "https://tuoitre.vn/canh-giac-chieu-tro-dua-di-nhat-trung-quoc-de-lam-dep-ho-chieu-roi-om-tien-bien-mat-100260916092755628.htm"),
    ]
    kb_df = pd.DataFrame(kb, columns=["scenario_id", "title", "content", "source_url"])
    kb_df.to_csv(DATA / "rag_knowledge_v1.csv", index=False, encoding="utf-8-sig")

    query_templates = {
        "K01_AUTHORITY": [
            "Có người xưng công an yêu cầu tôi chuyển tiền vào tài khoản tạm giữ để xác minh.",
            "Tin nhắn báo VNeID liên quan vụ án và bắt khai tài sản qua link.",
            "Người tự xưng tòa án dọa khóa hồ sơ nếu không nộp tiền ngay.",
            "Cán bộ gọi điện nói số điện thoại dính rửa tiền và yêu cầu giữ bí mật.",
            "Họ bảo nộp phạt nguội vào tài khoản cá nhân để xử lý nhanh.",
            "Người lạ mạo danh viện kiểm sát yêu cầu xác minh tài khoản ngân hàng.",
        ],
        "K02_BANK_PHISH": [
            "SMS ngân hàng báo giao dịch lạ rồi gửi link để hủy.",
            "Trang lạ yêu cầu nhập OTP để hoàn phí thẻ tín dụng.",
            "Tin nhắn bảo tài khoản ngân hàng sắp khóa và bắt đăng nhập đường dẫn.",
            "Thông báo nhận tiền nhưng phải bấm link xác nhận mới được cộng số dư.",
            "Người gọi tự xưng ngân hàng hỏi mật khẩu và mã OTP.",
            "Email ngân hàng giả yêu cầu cập nhật thông tin đăng nhập.",
        ],
        "K03_TASK": [
            "Làm nhiệm vụ chốt đơn phải nạp tiền trước rồi mới rút hoa hồng.",
            "Việc thả tim TikTok yêu cầu ứng tiền để mở cấp VIP.",
            "CTV đánh giá sản phẩm bị bắt nạp thêm tiền để hoàn tất đơn.",
            "Nhóm online cho rút nhỏ rồi yêu cầu nạp hàng chục triệu.",
            "Làm nhiệm vụ nhận hoa hồng nhưng hệ thống cứ bắt đóng thêm phí.",
            "Tuyển cộng tác viên sàn thương mại điện tử và bắt ứng vốn.",
        ],
        "K04_ACCOUNT_TAKEOVER": [
            "Bạn tôi nhắn mượn tiền nhưng tài khoản nhận lại là số lạ.",
            "Video call người thân bị đứng hình rồi họ xin chuyển tiền gấp.",
            "Tài khoản Facebook bạn bè nhắn vay tiền vì app ngân hàng lỗi.",
            "Người thân báo cấp cứu và yêu cầu chuyển vào tài khoản bác sĩ.",
            "Có cuộc gọi giống giọng con tôi nhưng thúc chuyển khoản ngay.",
            "Bạn cũ nhắn bằng tài khoản quen nhưng yêu cầu gửi tiền sang tài khoản mới.",
        ],
        "K05_PRIZE": [
            "Tôi được báo trúng quà nhưng phải đóng phí hồ sơ trước.",
            "Tin nhắn ZaloPay tặng quà yêu cầu đăng nhập trang lạ.",
            "Họ báo trúng xe máy và đòi phí thuế để nhận giải.",
            "Shipper nói có quà miễn phí nhưng phải trả khoản phí bất thường.",
            "Thông báo khách hàng may mắn bắt chuyển tiền trước khi nhận thưởng.",
            "Voucher lớn nhưng đường dẫn nhận quà không phải trang chính thức.",
        ],
        "K06_EXTORTION": [
            "Người lạ đe dọa phát tán clip riêng tư nếu tôi không chuyển tiền.",
            "Họ ghi lại video nhạy cảm rồi tống tiền qua mạng.",
            "Tin nhắn nói đã hack ảnh cá nhân và đòi tiền chuộc.",
            "Kẻ xấu dọa gửi hình cho đồng nghiệp nếu không thanh toán.",
            "Tống tiền bằng ảnh nhạy cảm và danh sách bạn bè.",
            "Đối tượng giả công an cô lập nạn nhân rồi ép gia đình chuyển tiền.",
        ],
        "K07_ECOM_REFUND": [
            "Người gọi biết đúng đơn hàng rồi bảo cài app để hoàn tiền.",
            "Shop giả nói sản phẩm cần thu hồi và gửi link nhận hoàn tiền.",
            "Nhân viên sàn yêu cầu OTP để hoàn tiền cho đơn cũ.",
            "Bên vận chuyển xin chia sẻ màn hình để xử lý khoản hoàn.",
            "Cuộc gọi hậu mãi dẫn tôi sang trang lạ để nhập tài khoản ngân hàng.",
            "Người tự xưng chăm sóc khách hàng yêu cầu thao tác ngân hàng vì đơn lỗi.",
        ],
        "K08_TRAVEL": [
            "Fanpage khách sạn giá rẻ yêu cầu cọc rồi chặn liên lạc.",
            "Email giả nền tảng đặt phòng gửi file lạ cho khách sạn.",
            "Người bán tour thanh lý giá rẻ bắt chuyển khoản toàn bộ trước.",
            "Trang giả khu nghỉ dưỡng yêu cầu đặt cọc vào tài khoản cá nhân.",
            "Đặt phòng qua fanpage nhưng sau khi chuyển tiền không liên lạc được.",
            "Email đặt phòng giả có link thanh toán đáng ngờ.",
        ],
        "K09_RECRUITMENT": [
            "Tin tuyển dụng bắt đóng phí đồng phục trước khi nhận việc.",
            "Công việc nhập liệu tại nhà yêu cầu cọc tiền thiết bị.",
            "Nhà tuyển dụng đòi phí hồ sơ để cấp mã nhân viên.",
            "Việc nhẹ lương cao nhưng phải chuyển tiền bảo đảm.",
            "Được báo trúng tuyển ngay và yêu cầu đóng phí đào tạo.",
            "Công ty lạ thu tiền trước khi gửi hợp đồng lao động.",
        ],
        "K10_TICKET": [
            "Mua vé concert qua tài khoản lạ, chuyển cọc xong bị chặn.",
            "Người bán vé sự kiện giá rẻ bắt thanh toán toàn bộ trước.",
            "Pass vé nhưng mã vé giả và người bán biến mất sau chuyển khoản.",
            "Tài khoản mới lập rao vé hiếm rẻ hơn thị trường.",
            "Đặt cọc vé xem ca nhạc nhưng không nhận được vé điện tử.",
            "Người bán dùng ảnh vé của người khác để yêu cầu chuyển tiền.",
        ],
        "K11_INVESTMENT": [
            "Người quen qua mạng rủ đầu tư sàn lợi nhuận cao và bắt nạp thêm tiền.",
            "Ứng dụng tiền số cho rút thử rồi khóa khi tôi nạp số tiền lớn.",
            "Sàn đầu tư lạ hứa lãi chắc chắn mỗi ngày.",
            "Người môi giới bảo vay tiền để nạp vào nền tảng đầu tư riêng.",
            "Tôi bị yêu cầu đóng phí để rút lợi nhuận từ sàn không rõ nguồn gốc.",
            "Tài khoản online khoe lợi nhuận và gửi link nạp tiền số.",
        ],
        "K12_VISA": [
            "Dịch vụ cam kết visa không cần phỏng vấn và bắt chuyển cọc vào tài khoản cá nhân.",
            "Môi giới nói làm đẹp hộ chiếu để xin visa rồi thu tiền trước.",
            "Người tự xưng nhân viên lãnh sự nhận tiền để cấp visa nhanh.",
            "Dịch vụ xuất khẩu lao động không hợp đồng nhưng yêu cầu đóng phí.",
            "Họ hứa chắc chắn đậu visa nếu nộp thêm phí quan hệ.",
            "Công ty du học chỉ liên hệ qua Zalo và bắt chuyển tiền trước khi xem hồ sơ.",
        ],
    }

    no_match = [
        "Tối nay nhà mình ăn gì?", "Lịch họp phòng đổi sang 9 giờ sáng.",
        "Tôi đã tự chuyển tiền giữa hai tài khoản của mình.", "Shop ơi đơn hàng của tôi giao ngày nào?",
        "Thầy cho em xin lịch nộp bài tập.", "Mai đội bóng đá tập ở sân cũ nhé.",
        "Tôi đã đặt vé tàu trên ứng dụng chính thức.", "Nhân sự gửi bảng lương trong cổng nội bộ.",
        "Mình đang đọc báo cáo tài chính để học đầu tư.", "Tôi đã đến ngân hàng đổi thẻ trực tiếp.",
        "Gia đình hẹn cuối tuần về quê.", "Công ty thông báo nghỉ lễ qua email nội bộ.",
        "Khách sạn xác nhận đặt phòng trong ứng dụng chính thức.", "Tôi gửi biên lai học phí qua cổng sinh viên.",
        "Mình đang so sánh giá laptop ở vài cửa hàng.", "Hôm qua nhóm ăn hết 600 nghìn và chia đều.",
        "Tôi tự tạo ticket hỗ trợ trên website chính thức.", "Lớp học chuyển phòng sang B203.",
        "Mẹ nhờ mua rau khi đi làm về.", "Tôi đã nhận khoản hoàn tiền trong ứng dụng ngân hàng.",
        "Nhóm nghiên cứu đang thảo luận về blockchain.", "Công ty gửi lịch đào tạo bảo mật nội bộ.",
        "Bạn tôi gửi ảnh chuyến du lịch hôm trước.", "Mình đặt lịch bảo hành máy tại trung tâm chính hãng.",
    ]

    dev_rows, test_rows = [], []
    for sid, qs in query_templates.items():
        for i, q in enumerate(qs):
            target = dev_rows if i < 3 else test_rows
            target.append({"query": q, "expected_scenario": sid, "query_type": "match"})
    for i, q in enumerate(no_match):
        target = dev_rows if i < 12 else test_rows
        target.append({"query": q, "expected_scenario": "NO_MATCH", "query_type": "no_match"})

    pd.DataFrame(dev_rows).to_csv(DATA / "rag_queries_dev.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(DATA / "rag_queries_test.csv", index=False, encoding="utf-8-sig")

    return {
        "knowledge_items": len(kb_df),
        "dev_queries": len(dev_rows),
        "test_queries": len(test_rows),
        "no_match_dev": sum(x["expected_scenario"] == "NO_MATCH" for x in dev_rows),
        "no_match_test": sum(x["expected_scenario"] == "NO_MATCH" for x in test_rows),
    }


def write_dataset_card(dev_info, ind_info, ood_info, rag_info):
    card = f"""# SafeNet Research Dataset Card v1\n\n## Freeze date\n2026-10-02\n\n## Development corpus\nThe original `dataset_vietscam_500_original.csv` contains 500 synthetic Vietnamese messages produced from fixed text templates. After exact-text deduplication, {dev_info['unique_rows']} usable unique messages remain. {dev_info['exact_duplicates_removed']} exact duplicate rows are removed. The surviving rows are mapped back to their originating template/scenario and split with `StratifiedGroupKFold`, so a template/scenario cannot appear in more than one of train, validation, or scenario-diagnostic test.\n\nThis development corpus has already been inspected during system development. Therefore it is **not** treated as an independent final test set.\n\n## Independent holdout v1\n`independent_test_v1.csv` contains {ind_info['rows']} newly authored synthetic messages ({ind_info['labels'].get(0,0)} benign and {ind_info['labels'].get(1,0)} scam) across {ind_info['scenarios']} scenarios. They were created only after Demo Freeze v1 and are not used for prompt, threshold, rule, or retrieval tuning. Exact-text overlap with the development corpus is zero.\n\nThe set is synthetic/author-curated rather than a naturally sampled population. Results must therefore be described as performance on a controlled independent holdout, not as population-level accuracy.\n\n## OOD/gray-area v1\n`ood_gray_v1.csv` contains {ood_info['rows']} samples across {ood_info['scenarios']} scenarios: unseen scam patterns, hard-negative benign messages, and deliberately under-specified gray-area messages. Gray-area rows use `expected_action=UNCERTAIN`; they are for selective-safety evaluation and are excluded from ordinary binary accuracy.\n\n## Retrieval benchmark\n`rag_knowledge_v1.csv` contains {rag_info['knowledge_items']} versioned scenario records. Retrieval model/threshold selection uses only `rag_queries_dev.csv`; `rag_queries_test.csv` remains untouched until the retrieval configuration is frozen. Test metrics are Recall@1, Recall@3, MRR, and false-retrieval rate on NO_MATCH queries.\n\n## Intended claims\nThe datasets support controlled comparative experiments and system ablation. They do not support claims of nationwide prevalence, deployment-grade detection accuracy, or demographic representativeness. A future journal extension should include a larger naturally sampled and independently annotated Vietnamese corpus.\n"""
    (DATA / "DATASET_CARD.md").write_text(card, encoding="utf-8")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    dev_info = prepare_development()
    ind_info = create_independent_test()
    ood_info = create_ood_gray()
    rag_info = create_rag_assets()
    write_dataset_card(dev_info, ind_info, ood_info, rag_info)

    info = {
        "freeze_date": "2026-10-02",
        "seed": SEED,
        "development": dev_info,
        "independent": ind_info,
        "ood_gray": ood_info,
        "rag": rag_info,
    }
    (DATA / "data_preparation_summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    manifests = {}
    for p in sorted(DATA.iterdir()):
        if p.is_file():
            manifests[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    (DATA / "DATA_SHA256.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")

    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("Prepared files:")
    for p in sorted(DATA.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
