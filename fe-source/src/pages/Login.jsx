import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";

export default function Login() {
  const navigate = useNavigate();
  const [account, setAccount] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState({ type: "", message: "" });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!status.message) {
      return;
    }
    const timer = setTimeout(() => {
      setStatus({ type: "", message: "" });
    }, 2000);
    return () => clearTimeout(timer);
  }, [status.message]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setStatus({ type: "", message: "" });

    if (!account.trim() || !password) {
      setStatus({ type: "error", message: "Vui lòng nhập đủ tài khoản và mật khẩu." });
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account: account.trim(), password }),
      });

      const data = await res.json();

      if (!res.ok) {
        const detail = (data && data.detail) || "";
        if (detail === "account is empty" || detail === "account not found") {
          setStatus({ type: "error", message: "Tài khoản không tồn tại." });
        } else if (detail === "invalid password") {
          setStatus({ type: "error", message: "Sai mật khẩu." });
        } else {
          setStatus({ type: "error", message: "Đăng nhập thất bại. Vui lòng thử lại." });
        }
        return;
      }

      localStorage.setItem("user_id", data.user_id);
      localStorage.setItem("account", data.account);
      setStatus({
        type: "success",
        message: "Đăng nhập thành công.",
      });
      navigate("/menu");
    } catch (err) {
      setStatus({ type: "error", message: "Không kết nối được server." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-mesh bg-no-repeat bg-cover grain">
      <div className="mx-auto flex min-h-screen max-w-6xl items-center px-6 py-16">
        <div className="grid w-full gap-12 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-6">
            <p className="text-sm uppercase tracking-[0.35em] text-haze">
              Chat Lis Speak
            </p>
            <h1 className="text-4xl font-semibold text-fog md:text-5xl">
              Luyện nghe nói tiếng Anh theo nhịp của bạn.
            </h1>
            <p className="max-w-xl text-base text-haze md:text-lg">
              Đăng nhập để bắt đầu học với các buổi luyện tập ngắn, gợi ý thông
              minh và phản hồi rõ ràng.
            </p>
            <div className="flex flex-wrap gap-3">
              <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-fog">
                Luyện nghe
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-fog">
                Luyện nói
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-fog">
                Voice-friendly
              </span>
            </div>
          </div>
          <div className="glass rounded-3xl p-8 shadow-soft">
            <h2 className="text-2xl font-semibold text-fog">Đăng nhập</h2>
            <p className="mt-2 text-sm text-haze">
              Sử dụng tài khoản nội bộ hoặc email.
            </p>
            <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
              <label className="block text-sm text-haze">
                Email
                <input
                  type="email"
                  placeholder="ban@email.com"
                  value={account}
                  onChange={(e) => setAccount(e.target.value)}
                  className="mt-2 w-full rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-fog outline-none focus:border-tide"
                />
              </label>
              <label className="block text-sm text-haze">
                Mật khẩu
                <input
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="mt-2 w-full rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-fog outline-none focus:border-ember"
                />
              </label>
              <div className="flex items-center justify-between text-xs text-haze">
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="accent-tide" />
                  Ghi nhớ đăng nhập
                </label>
                <button type="button" className="text-fog underline">
                  Quên mật khẩu?
                </button>
              </div>
              <button
                type="submit"
                disabled={loading}
                className="mt-2 w-full rounded-2xl bg-tide px-5 py-3 text-center font-medium text-white shadow-soft hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Đang đăng nhập..." : "Vào học ngay"}
              </button>
            </form>
            {status.message ? (
              <div
                className={`mt-6 rounded-2xl border p-4 text-sm ${
                  status.type === "success"
                    ? "border-pine/40 bg-pine/10 text-fog"
                    : "border-ember/40 bg-ember/10 text-fog"
                }`}
              >
                {status.message}
              </div>
            ) : (
              <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-haze">
                Chưa có tài khoản?{" "}
                <Link to="/signup" className="text-fog underline">
                  Tạo tài khoản
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
