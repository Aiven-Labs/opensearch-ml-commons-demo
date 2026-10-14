const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const runButton = document.querySelector("#run-button");
const runState = document.querySelector("#run-state");
const errorPanel = document.querySelector("#error-panel");
const errorMessage = document.querySelector("#error-message");
const answerSection = document.querySelector("#answer-section");
const retrievalSection = document.querySelector("#retrieval-section");
const memoryLabel = document.querySelector("#memory-label");
const resetMemoryButton = document.querySelector("#reset-memory");
const cliCommand = document.querySelector("#cli-command");
const copyCommandButton = document.querySelector("#copy-command");
const compareRetrieval = document.querySelector("#compare-retrieval");

let memoryId = sessionStorage.getItem("opensearch-demo-memory-id");
let memoryAgentId = sessionStorage.getItem("opensearch-demo-memory-agent-id");

function renderMemoryState() {
  if (memoryId) {
    memoryLabel.textContent = `Continuing memory ${memoryId.slice(0, 8)}…`;
    resetMemoryButton.hidden = false;
  } else {
    memoryLabel.textContent = "New conversation";
    resetMemoryButton.hidden = true;
  }
}

function shellQuote(value) {
  return `'${value.replaceAll("'", `'\"'\"'`)}'`;
}

function updateCliCommand() {
  cliCommand.textContent =
    "python3 scripts/deterministic_agent.py ask \\\n" +
    `  --question ${shellQuote(questionInput.value.trim())}`;
}

function setLoading(loading) {
  runButton.disabled = loading;
  runButton.textContent = loading ? "Running…" : "Run agent";
  runState.hidden = !loading;
  form.setAttribute("aria-busy", String(loading));
}

function resultCard(result) {
  const article = document.createElement("article");
  article.className = "result";

  const top = document.createElement("div");
  top.className = "result__top";
  const rank = document.createElement("span");
  rank.className = "result__rank";
  rank.textContent = `Rank ${result.rank}`;
  const score = document.createElement("span");
  score.className = "result__score";
  score.textContent = `score ${Number(result.score).toFixed(4)}`;
  top.append(rank, score);

  const title = document.createElement("h4");
  title.textContent = result.title;
  const excerpt = document.createElement("p");
  excerpt.textContent = result.text;
  const source = document.createElement("a");
  source.href = result.url;
  source.target = "_blank";
  source.rel = "noopener noreferrer";
  source.textContent = "Open source documentation ↗";

  article.append(top, title, excerpt, source);
  return article;
}

function renderResults(targetId, results) {
  const target = document.querySelector(targetId);
  target.replaceChildren(...results.map(resultCard));
}

function renderResponse(data) {
  const usage = data.agent.token_usage || {};
  document.querySelector("#answer-text").textContent =
    data.agent.answer || "No answer returned.";
  document.querySelector("#model-name").textContent = data.agent.model || "—";
  document.querySelector("#input-tokens").textContent =
    usage.prompt_tokens ?? "—";
  document.querySelector("#output-tokens").textContent =
    usage.completion_tokens ?? "—";
  document.querySelector("#agent-time").textContent =
    `${data.timings_ms.agent_execution} ms`;
  document.querySelector("#total-time").textContent =
    `${data.timings_ms.total} ms total`;

  if (data.retrieval) {
    renderResults("#keyword-results", data.retrieval.keyword);
    renderResults("#vector-results", data.retrieval.vector);
    renderResults("#hybrid-results", data.retrieval.hybrid);
    document.querySelector("#retrieval-time").textContent =
      `${data.timings_ms.retrieval_comparison} ms`;
    retrievalSection.hidden = false;
  } else {
    document.querySelector("#retrieval-time").textContent =
      "First-turn comparison retained";
  }

  memoryId = data.agent.memory_id || memoryId;
  if (memoryId) {
    sessionStorage.setItem("opensearch-demo-memory-id", memoryId);
    memoryAgentId = data.agent_id;
    sessionStorage.setItem("opensearch-demo-memory-agent-id", memoryAgentId);
  }
  renderMemoryState();
  answerSection.hidden = false;
  answerSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function askAgent(question) {
  setLoading(true);
  errorPanel.hidden = true;
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        memory_id: memoryId,
        include_retrieval_comparison: compareRetrieval.checked && !memoryId,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    renderResponse(data);
  } catch (error) {
    errorMessage.textContent = error.message;
    errorPanel.hidden = false;
    errorPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } finally {
    setLoading(false);
  }
}

async function loadStatus() {
  const status = document.querySelector(".status");
  const label = document.querySelector("#cluster-status");
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Status unavailable");
    if (memoryId && memoryAgentId !== data.agent_id) {
      memoryId = null;
      memoryAgentId = null;
      sessionStorage.removeItem("opensearch-demo-memory-id");
      sessionStorage.removeItem("opensearch-demo-memory-agent-id");
      renderMemoryState();
    }
    status.classList.add("status--ready");
    label.textContent = `${data.documents} docs · agent ready`;
  } catch (error) {
    label.textContent = "Cluster unavailable";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) askAgent(question);
});

questionInput.addEventListener("input", updateCliCommand);

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.requiresMemory === "true" && !memoryId) {
      errorMessage.textContent =
        "Run an initial question first. This follow-up requires conversation memory.";
      errorPanel.hidden = false;
      errorPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    questionInput.value = button.dataset.question;
    updateCliCommand();
    questionInput.focus();
  });
});

resetMemoryButton.addEventListener("click", () => {
  memoryId = null;
  memoryAgentId = null;
  sessionStorage.removeItem("opensearch-demo-memory-id");
  sessionStorage.removeItem("opensearch-demo-memory-agent-id");
  renderMemoryState();
});

copyCommandButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(cliCommand.textContent);
    copyCommandButton.textContent = "Copied";
    setTimeout(() => {
      copyCommandButton.textContent = "Copy command";
    }, 1_200);
  } catch {
    copyCommandButton.textContent = "Select and copy";
  }
});

updateCliCommand();
renderMemoryState();
loadStatus();
