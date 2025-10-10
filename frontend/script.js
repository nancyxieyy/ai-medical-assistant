const appShell = document.querySelector(".app-shell");
const chatPane = document.querySelector("#chat-pane");
const collapseToggle = document.querySelector("#chat-collapse-toggle");
const conversationThread = document.querySelector("#conversation-thread");
const composerForm = document.querySelector("#composer-form");
const composerInput = document.querySelector("#composer-input");

const createThreadMessage = (content, variant = "outgoing") => {
  const article = document.createElement("article");
  article.className = `thread-message thread-message--${variant}`;

  const paragraph = document.createElement("p");
  paragraph.textContent = content;

  article.append(paragraph);
  return article;
};

const appendMessage = (element) => {
  if (!conversationThread) {
    return;
  }

  conversationThread.append(element);
  conversationThread.scrollTop = conversationThread.scrollHeight;
};

composerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = composerInput.value.trim();
  if (!text) {
    return;
  }

  appendMessage(createThreadMessage(text));
  composerInput.value = "";
  autoResizeTextarea();
  composerInput.focus();
});

const autoResizeTextarea = () => {
  composerInput.style.height = "auto";
  composerInput.style.height = `${composerInput.scrollHeight}px`;
};

composerInput.addEventListener("input", autoResizeTextarea);
autoResizeTextarea();

// 回车发送，Shift+回车换行
composerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composerForm.requestSubmit();
  }
});

window.appendIncomingMessage = (content) => {
  const text = typeof content === "string" ? content.trim() : "";

  if (!text) {
    return;
  }

  appendMessage(createThreadMessage(text, "incoming"));
};

if (collapseToggle && appShell && chatPane) {
  chatPane.setAttribute("aria-hidden", "false");

  const toggleChatVisibility = () => {
    const collapsed = appShell.classList.toggle("chat-collapsed");
    collapseToggle.setAttribute("aria-expanded", String(!collapsed));
    collapseToggle.textContent = collapsed ? ">>" : "<<";
    chatPane.setAttribute("aria-hidden", collapsed ? "true" : "false");
  };

  collapseToggle.addEventListener("click", toggleChatVisibility);
}

// History pane collapse functionality
const historyPane = document.querySelector("#history-pane");
const historyCollapseToggle = document.querySelector("#history-collapse-toggle");

if (historyCollapseToggle && appShell && historyPane) {
  historyPane.setAttribute("aria-hidden", "false");

  const toggleHistoryVisibility = () => {
    const collapsed = appShell.classList.toggle("history-collapsed");
    historyCollapseToggle.setAttribute("aria-expanded", String(!collapsed));
    historyCollapseToggle.textContent = collapsed ? "<<" : ">>";
    historyPane.setAttribute("aria-hidden", collapsed ? "true" : "false");
  };

  historyCollapseToggle.addEventListener("click", toggleHistoryVisibility);
}

// Medical records expand/collapse functionality
const recordItems = document.querySelectorAll(".record-item");

recordItems.forEach((item) => {
  item.addEventListener("click", () => {
    item.classList.toggle("expanded");
  });
});

// Summary modal functionality
const summaryToggle = document.querySelector("#summary-toggle");
const summaryModal = document.querySelector("#summary-modal");
const summaryEditor = document.querySelector("#summary-editor");
const summarySave = document.querySelector("#summary-save");
const confirmModal = document.querySelector("#confirm-modal");
const confirmOk = document.querySelector("#confirm-ok");
const confirmEdit = document.querySelector("#confirm-edit");

summaryToggle?.addEventListener("click", () => {
  summaryModal?.classList.add("active");
});

summarySave?.addEventListener("click", () => {
  summaryModal?.classList.remove("active");
  confirmModal?.classList.add("active");
});

confirmOk?.addEventListener("click", () => {
  confirmModal?.classList.remove("active");
});

confirmEdit?.addEventListener("click", () => {
  confirmModal?.classList.remove("active");
  summaryModal?.classList.add("active");
});

summaryModal?.addEventListener("click", (e) => {
  if (e.target === summaryModal) {
    summaryModal.classList.remove("active");
  }
});

confirmModal?.addEventListener("click", (e) => {
  if (e.target === confirmModal) {
    confirmModal.classList.remove("active");
  }
});

