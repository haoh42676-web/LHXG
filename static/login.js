const form = document.querySelector("#loginForm");
const msg = document.querySelector("#loginMessage");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  msg.textContent = "正在验证";
  const payload = {
    username: document.querySelector("#username").value.trim(),
    password: document.querySelector("#password").value.trim(),
  };
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      msg.textContent = "账号或密码错误";
      return;
    }
    window.location.href = "/";
  } catch (error) {
    msg.textContent = `登录失败：${error}`;
  }
});
