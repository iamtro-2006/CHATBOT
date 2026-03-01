import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Signup() {
  const navigate = useNavigate();
  const [account, setAccount] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [status, setStatus] = useState({ type: "", message: "" });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setStatus({ type: "", message: "" });

    if (!account.trim() || !password) {
      setStatus({ type: "error", message: "Vui lòng nhập đủ thông tin." });
      return;
    }
    if (password !== confirm) {
      setStatus({ type: "error", message: "Mật khẩu xác nhận không khớp." });
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account: account.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = (data && data.detail) || "";
        if (detail === "account already exists") {
          setStatus({ type: "error", message: "Tài khoản đã tồn tại." });
        } else {
          setStatus({ type: "error", message: "Tạo tài khoản thất bại." });
        }
        return;
      }

      localStorage.setItem("user_id", data.user_id);
      localStorage.setItem("account", data.account);
      setStatus({ type: "success", message: "Tạo tài khoản thành công." });
      navigate("/menu");
    } catch (err) {
      setStatus({ type: "error", message: "Không kết nối được server." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-mesh bg-no-repeat bg-cover grain">
      <div className="mx-auto flex min-h-screen max-w-5xl items-center px-6 py-16">
        <div className="glass w-full rounded-3xl p-8 shadow-soft">
          <h1 className="text-3xl font-semibold text-fog">Tạo tài khoản</h1>
          <p className="mt-2 text-sm text-haze">
            Tạo tài khoản mới để bắt đầu luyện tập.
          </p>

          <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
            <label className="block text-sm text-haze">
              Email / Account
              <input
                type="text"
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
            <label className="block text-sm text-haze">
              Xác nhận mật khẩu
              <input
                type="password"
                placeholder="••••••••"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-fog outline-none focus:border-ember"
              />
            </label>

            <button
              type="submit"
              disabled={loading}
              className="mt-2 w-full rounded-2xl bg-ember px-5 py-3 text-center font-medium text-white shadow-soft hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Đang tạo..." : "Tạo tài khoản"}
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
          ) : null}

          <div className="mt-6 text-sm text-haze">
            Đã có tài khoản?{" "}
            <Link to="/login" className="text-fog underline">
              Đăng nhập
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
