import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";

export default function Menu() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState({
    completed_days: 0,
    current_day: 1,
  });
  const [listening, setListening] = useState(false);
  const [voiceText, setVoiceText] = useState("");
  const [speechSupported, setSpeechSupported] = useState(true);

  const recogRef = useRef(null);
  const finalTranscriptRef = useRef("");
  const silenceTimerRef = useRef(null);
  const startTriggeredRef = useRef(false);
  const listeningWantedRef = useRef(false);
  const validatingRef = useRef(false);
  const voiceTextRef = useRef("");

  const clearSilenceTimer = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  };
  const clearTranscript = () => {
    finalTranscriptRef.current = "";
    voiceTextRef.current = "";
    setVoiceText("");
  };

  const createThreadAndNavigate = () => {
    if (startTriggeredRef.current) return;
    startTriggeredRef.current = true;

    const threadId =
      (typeof crypto !== "undefined" && crypto.randomUUID && crypto.randomUUID()) ||
      `thread_${Date.now()}_${Math.random().toString(16).slice(2)}`;

    localStorage.setItem("thread_id", threadId);
    navigate("/conversation");
  };

  const stopListening = () => {
    listeningWantedRef.current = false;
    clearSilenceTimer();

    if (recogRef.current) {
      try {
        recogRef.current.stop();
      } catch {
        // ignore
      }
    }
    setListening(false);
  };

  const startListening = () => {
    if (!recogRef.current) return;
    listeningWantedRef.current = true;
    try {
      recogRef.current.start();
      setListening(true);
    } catch {
      // ignore rapid start errors
    }
  };

  const validateVoiceIntent = async (text) => {
    if (validatingRef.current || startTriggeredRef.current) return;

    const userId = localStorage.getItem("user_id");
    if (!userId) {
      navigate("/login");
      return;
    }

    const payloadText = (text || "").trim();
    if (!payloadText) return;

    validatingRef.current = true;
    try {
      const res = await fetch("http://localhost:8000/validate-intent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          message: payloadText,
        }),
      });

      if (!res.ok) return;

      const data = await res.json();
      if (data?.should_start) {
        stopListening();
        createThreadAndNavigate();
      }
    } catch {
      // ignore connection errors for continuous voice mode
    } finally {
      validatingRef.current = false;
    }
  };

  useEffect(() => {
    const userId = localStorage.getItem("user_id");
    if (!userId) {
      return;
    }
    const controller = new AbortController();
    const loadProgress = async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/progress?user_id=${encodeURIComponent(userId)}`,
          { signal: controller.signal }
        );
        if (!res.ok) {
          return;
        }
        const data = await res.json();
        setProgress({
          completed_days: Number(data.completed_days || 0),
          current_day: Number(data.current_day || 1),
        });
      } catch (err) {
        if (err.name !== "AbortError") {
          return;
        }
      }
    };
    loadProgress();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    voiceTextRef.current = voiceText;
  }, [voiceText]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
      return;
    }

    const recog = new SpeechRecognition();
    recog.lang = "en-US";
    recog.interimResults = true;
    recog.continuous = true;
    recog.maxAlternatives = 1;

    recog.onresult = (event) => {
      let final = "";
      let interim = "";

      for (let i = 0; i < event.results.length; i += 1) {
        const chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += chunk;
        else interim += chunk;
      }

      const merged = `${final}${interim}`.trim();
      if (!merged) return;

      if (final) {
        finalTranscriptRef.current = final.trim();
      }
      setVoiceText(merged);

      clearSilenceTimer();
      silenceTimerRef.current = setTimeout(() => {
        const textToValidate = (finalTranscriptRef.current || voiceTextRef.current || "").trim();
        if (textToValidate) {
          void validateVoiceIntent(textToValidate);
        }
        clearTranscript();
      }, 2000);
    };

    recog.onend = () => {
      setListening(false);
      if (listeningWantedRef.current && !startTriggeredRef.current) {
        startListening();
      }
    };

    recog.onerror = () => {
      setListening(false);
      if (listeningWantedRef.current && !startTriggeredRef.current) {
        startListening();
      }
    };

    recogRef.current = recog;
    startListening();

    return () => {
      clearSilenceTimer();
      listeningWantedRef.current = false;
      try {
        recog.stop();
      } catch {
        // ignore
      }
    };
  }, []);

  const isNewUser = progress.completed_days === 0;
  const highlightUpTo = isNewUser ? 0 : Math.min(progress.completed_days + 1, 4);

  const handleLogout = () => {
    stopListening();
    localStorage.removeItem("user_id");
    localStorage.removeItem("account");
    localStorage.removeItem("thread_id");
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-ink text-fog">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-haze">Dashboard</p>
            <h1 className="text-3xl font-semibold md:text-4xl">Menu học tập</h1>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-haze">
              English Coach · Level A1
            </div>
            <button
              onClick={handleLogout}
              className="rounded-full border border-ember/50 bg-ember/10 px-4 py-2 text-sm text-ember"
            >
              Đăng xuất
            </button>
          </div>
        </header>

        <section className="mt-10 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="glass rounded-3xl p-8 shadow-soft">
            <h2 className="text-2xl font-semibold">Bắt đầu học</h2>
            <p className="mt-2 text-sm text-haze">
              Bạn có thể nói tiếng Anh như: "start", "let&apos;s begin", "I want to learn now".
            </p>
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-fog">
              MIC State: {speechSupported ? (listening ? "ON" : "OFF") : "Not supported"}
            </div>
            <div className="mt-3 rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm text-haze">
              Transcript: {voiceText || "..."}
            </div>
            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              {[
                { title: "Giao tiếp hằng ngày", desc: "Chào hỏi, mua sắm, hỏi đường" },
                { title: "Du lịch", desc: "Sân bay, khách sạn, nhà hàng" },
                { title: "Công việc", desc: "Email, họp nhóm, báo cáo" },
                { title: "Học tập", desc: "Thuyết trình, hỏi bài, thảo luận" },
                { title: "Sức khỏe", desc: "Bác sĩ, thuốc, tình trạng" },
                { title: "Giải trí", desc: "Phim ảnh, sở thích, bạn bè" },
              ].map((item) => (
                <div
                  key={item.title}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3"
                >
                  <p className="text-sm font-semibold text-fog">{item.title}</p>
                  <p className="mt-1 text-xs text-haze">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>

          <aside className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-white/10 via-white/5 to-transparent p-6">
              <h3 className="text-lg font-semibold">Chuỗi ngày học</h3>
              <p className="mt-2 text-sm text-haze">
                4 ngày luyện tập · {progress.completed_days} ngày đã hoàn thành
              </p>
              <div className="mt-4 grid grid-cols-4 gap-2">
                {[1, 2, 3, 4].map((day) => (
                  <div
                    key={day}
                    className={`h-12 rounded-2xl border text-center text-xs leading-[3rem] ${
                      day <= highlightUpTo
                        ? "border-tide bg-tide/20 text-fog"
                        : isNewUser && day === 1
                        ? "border-tide/60 bg-tide/10 text-fog font-semibold"
                        : "border-tide/40 bg-transparent text-fog"
                    }`}
                  >
                    Ngày {day}
                  </div>
                ))}
              </div>
            </div>
            <div className="glass rounded-3xl p-6 shadow-soft">
              <h3 className="text-lg font-semibold">Ghi chú nhanh</h3>
              <ul className="mt-4 space-y-3 text-sm text-haze">
                <li>• Ưu tiên luyện nghe 10 phút mỗi ngày.</li>
                <li>• Chủ đề mới sẽ làm sau khi bạn chọn.</li>
                <li>• Bạn có thể nói lệnh bắt đầu thay vì bấm nút.</li>
              </ul>
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
}
