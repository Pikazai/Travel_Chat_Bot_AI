"""
CSS styles for Streamlit UI.
"""
STYLES = """
<style>
:root{
  --primary:#2b4c7e;
  --accent:#e7f3ff;
  --muted:#f2f6fa;
}
body {
  background: linear-gradient(90deg, #f8fbff 0%, #eef5fa 100%);
  font-family: 'Segoe UI', Roboto, Arial, sans-serif;
}
.stApp > header {visibility: hidden;}
h1, h2, h3 { color: var(--primary); }
.sidebar-card { background-color:#f1f9ff; padding:10px; border-radius:10px; margin-bottom:8px;}
.user-message { background: #f2f2f2; padding:10px; border-radius:12px; }
.assistant-message { background: #e7f3ff; padding:10px; border-radius:12px; }
.pill-btn { border-radius:999px !important; background:#e3f2fd !important; color:var(--primary) !important; padding:6px 12px; border: none; }
.status-ok { background:#d4edda; padding:8px; border-radius:8px; }
.status-bad { background:#f8d7da; padding:8px; border-radius:8px; }
.small-muted { color: #6b7280; font-size:12px; }
.logo-title { display:flex; align-items:center; gap:10px; }
.logo-title h1 { margin:0; }
.assistant-bubble {
  background-color: #e7f3ff;
  padding: 12px 16px;
  border-radius: 15px;
  margin-bottom: 6px;
}
.user-message {
  background-color: #f2f2f2;
  padding: 12px 16px;
  border-radius: 15px;
  margin-bottom: 6px;
}
/* HERO */
.hero {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 8px 30px rgba(43,76,126,0.12);
  margin-bottom: 18px;
}
.hero__bg {
  width: 100%;
  height: 320px;
  object-fit: cover;
  filter: brightness(0.65) saturate(1.05);
}
.hero__overlay {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.hero__card {
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.08));
  backdrop-filter: blur(6px);
  border-radius: 12px;
  padding: 18px;
  width: 100%;
  max-width: 980px;
  color: white;
}
.hero__title { font-size: 28px; font-weight:700; margin:0 0 6px 0; color: #fff; }
.hero__subtitle { margin:0 0 12px 0; color: #f0f6ff; }
.hero__cta { display:flex; gap:8px; align-items:center; }
@media (max-width: 768px) {
  .hero__bg { height: 220px; }
  .hero__title { font-size: 20px; }
}
.audio-wrapper {margin-top: 6px;}
</style>
"""

