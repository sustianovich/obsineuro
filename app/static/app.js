const elements = {
    desktop: document.querySelector(".desktop"),
    themeToggle: document.getElementById("theme-toggle"),
    sidebar: document.querySelector(".sidebar"),
    sidebarAdvancedToggle: document.getElementById(
        "sidebar-advanced-toggle"
    ),
    mobileTabs: Array.from(document.querySelectorAll(".mobile-tab")),
    workspace: document.getElementById("workspace"),
    welcome: document.getElementById("welcome"),
    conversationHeader: document.getElementById("conversation-header"),
    conversationTitle: document.getElementById("conversation-title"),
    chatThread: document.getElementById("chat-thread"),
    conversationList: document.getElementById("conversation-list"),
    conversationListEmpty: document.getElementById(
        "conversation-list-empty"
    ),
    projectList: document.getElementById("project-list"),
    projectListEmpty: document.getElementById("project-list-empty"),
    newProjectButton: document.getElementById("new-project-button"),
    deleteProjectButton: document.getElementById("delete-project-button"),
    projectAgentSettings: document.getElementById(
        "project-agent-settings"
    ),
    projectVerificationEnabled: document.getElementById(
        "project-verification-enabled"
    ),
    projectMemoryEnabled: document.getElementById(
        "project-memory-enabled"
    ),
    verifierContextTokens: document.getElementById(
        "verifier-context-tokens"
    ),
    writerContextTokens: document.getElementById(
        "writer-context-tokens"
    ),
    verifierContextValue: document.getElementById(
        "verifier-context-value"
    ),
    writerContextValue: document.getElementById(
        "writer-context-value"
    ),
    verifierContextUsage: document.getElementById(
        "verifier-context-usage"
    ),
    writerContextUsage: document.getElementById(
        "writer-context-usage"
    ),
    verifierContextBar: document.getElementById(
        "verifier-context-bar"
    ),
    writerContextBar: document.getElementById("writer-context-bar"),
    projectContextOverall: document.getElementById(
        "project-context-overall"
    ),
    projectMemorySummaryStatus: document.getElementById(
        "project-memory-summary-status"
    ),
    saveProjectAgentSettings: document.getElementById(
        "save-project-agent-settings"
    ),
    workspaceActiveProject: document.getElementById(
        "workspace-active-project"
    ),
    newConversationButton: document.getElementById(
        "new-conversation-button"
    ),
    indexButton: document.getElementById("index-button"),
    selectVaultButton: document.getElementById("select-vault-button"),
    chatProfile: document.getElementById("chat-profile"),
    customChatProfile: document.getElementById("custom-chat-profile"),
    documentCount: document.getElementById("document-count"),
    chunkCount: document.getElementById("chunk-count"),
    vaultPath: document.getElementById("vault-path"),
    systemMessage: document.getElementById("system-message"),
    form: document.getElementById("question-form"),
    question: document.getElementById("question"),
    askButton: document.getElementById("ask-button"),
    stopQueryButton: document.getElementById("stop-query-button"),
    queryStatus: document.getElementById("query-status"),
    topK: document.getElementById("top-k"),
    statusFilter: document.getElementById("status-filter"),
    vigenciaFilter: document.getElementById("vigencia-filter"),
    tagFilter: document.getElementById("tag-filter"),
    tagFilterControl: document.getElementById("tag-filter-control"),
    tagFilterSummary: document.getElementById("tag-filter-summary"),
    tagFilterSearch: document.getElementById("tag-filter-search"),
    tagFilterClear: document.getElementById("tag-filter-clear"),
    tagFilterEmpty: document.getElementById("tag-filter-empty"),
    contextCeilingHint: document.getElementById("context-ceiling-hint"),
    historyDrawer: document.getElementById("history-drawer"),
    expandLinks: document.getElementById("expand-links"),
    expandLinksLabel: document.getElementById("expand-links-label"),
    useMemory: document.getElementById("use-memory"),
    memoryStatus: document.getElementById("memory-status"),
    textDialog: document.getElementById("text-dialog"),
    textDialogTitle: document.getElementById("text-dialog-title"),
    textDialogLabel: document.getElementById("text-dialog-label"),
    textDialogInput: document.getElementById("text-dialog-input"),
    textDialogConfirm: document.getElementById("text-dialog-confirm"),
    confirmDialog: document.getElementById("confirm-dialog"),
    confirmDialogTitle: document.getElementById("confirm-dialog-title"),
    confirmDialogMessage: document.getElementById(
        "confirm-dialog-message"
    ),
    confirmDialogConfirm: document.getElementById(
        "confirm-dialog-confirm"
    ),
    contextIndicator: document.getElementById("context-indicator"),
    contextIndicatorPercent: document.getElementById(
        "context-indicator-percent"
    ),
    contextIndicatorModel: document.getElementById(
        "context-indicator-model"
    ),
    contextIndicatorWindow: document.getElementById(
        "context-indicator-window"
    ),
    contextInspectorDialog: document.getElementById(
        "context-inspector-dialog"
    ),
    ciModelName: document.getElementById("ci-model-name"),
    ciModelProfile: document.getElementById("ci-model-profile"),
    ciModelCapacity: document.getElementById("ci-model-capacity"),
    ciModelSource: document.getElementById("ci-model-source"),
    ciModelVerified: document.getElementById("ci-model-verified"),
    ciModelWarning: document.getElementById("ci-model-warning"),
    ciConversationEmpty: document.getElementById("ci-conversation-empty"),
    ciConversationGrid: document.getElementById("ci-conversation-grid"),
    ciTotalTurns: document.getElementById("ci-total-turns"),
    ciStoredTokens: document.getElementById("ci-stored-tokens"),
    ciRecentTurns: document.getElementById("ci-recent-turns"),
    ciRecentTokens: document.getElementById("ci-recent-tokens"),
    ciSummaryTokens: document.getElementById("ci-summary-tokens"),
    ciProjectMemoryTokens: document.getElementById(
        "ci-project-memory-tokens"
    ),
    ciNextEstimate: document.getElementById("ci-next-estimate"),
    ciVerifier: document.getElementById("ci-verifier"),
    ciWriter: document.getElementById("ci-writer"),
    ciWarnings: document.getElementById("ci-warnings"),
    turnInspectorDialog: document.getElementById("turn-inspector-dialog"),
    turnInspectorModel: document.getElementById("turn-inspector-model"),
    turnInspectorVerifier: document.getElementById(
        "turn-inspector-verifier"
    ),
    turnInspectorWriter: document.getElementById("turn-inspector-writer"),
};

let activeConversationId = null;
let activeConversationTitle = "";
let activeProjectId = null;
let activeProjectName = "";
let defaultProjectId = "default";
let loadedProjects = [];
let pendingTurnNumber = 0;
let activeQueryController = null;
const ACTIVE_PROJECT_KEY = "obsidian-rag-active-project";
const MEMORY_PREFERENCE_KEY = "obsidian-rag-use-memory";
const THEME_KEY = "obsidian-rag-theme";
let memorySummaryInterval = 10;
let memoryRecentTurns = 4;
let lastStatusData = null;
let latestTurnMaxUsagePercent = null;

async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
        headers: {"Content-Type": "application/json"},
        ...options,
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.detail || "Se produjo un error inesperado.");
    }
    return payload;
}

function setMobileTab(tab) {
    const active = ["chat", "status", "projects", "filters"].includes(tab)
        ? tab
        : "chat";
    if (elements.desktop) {
        elements.desktop.dataset.mobileTab = active;
    }
    for (const button of elements.mobileTabs) {
        const selected = button.dataset.mobileTab === active;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", String(selected));
    }
}

setMobileTab("chat");

for (const button of elements.mobileTabs) {
    button.addEventListener("click", () => {
        setMobileTab(button.dataset.mobileTab);
    });
}

function setTheme(theme) {
    const dark = theme === "dark";
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    if (elements.themeToggle) {
        const label = dark
            ? "Activar fondo claro"
            : "Activar fondo oscuro";
        elements.themeToggle.textContent = dark ? "Claro" : "Oscuro";
        elements.themeToggle.setAttribute("aria-pressed", String(dark));
        elements.themeToggle.title = label;
        elements.themeToggle.setAttribute("aria-label", label);
    }
    window.localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
}

setTheme(
    window.localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light"
);

elements.themeToggle?.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark";
    setTheme(dark ? "light" : "dark");
});

// El bloque "Agentes" (ajuste fino de contexto) queda oculto por defecto
// para no saturar la barra lateral: es un ajuste de uso ocasional, no el
// flujo principal de preguntar. Un toggle lo trae de vuelta cuando hace
// falta, y la preferencia se recuerda entre sesiones.
const SIDEBAR_ADVANCED_KEY = "obsidian-rag-sidebar-advanced";

function setSidebarAdvanced(enabled) {
    if (elements.sidebar) {
        elements.sidebar.dataset.advanced = String(enabled);
    }
    if (elements.sidebarAdvancedToggle) {
        const label = enabled
            ? "Ocultar ajustes avanzados"
            : "Mostrar ajustes avanzados";
        elements.sidebarAdvancedToggle.setAttribute(
            "aria-pressed",
            String(enabled)
        );
        elements.sidebarAdvancedToggle.title = label;
        elements.sidebarAdvancedToggle.setAttribute("aria-label", label);
    }
    window.localStorage.setItem(SIDEBAR_ADVANCED_KEY, String(enabled));
}

setSidebarAdvanced(
    window.localStorage.getItem(SIDEBAR_ADVANCED_KEY) === "true"
);

elements.sidebarAdvancedToggle?.addEventListener("click", () => {
    const enabled =
        elements.sidebarAdvancedToggle.getAttribute("aria-pressed")
        !== "true";
    setSidebarAdvanced(enabled);
});

function supportsDialog(dialog) {
    return dialog && typeof dialog.showModal === "function";
}

function isAbortError(error) {
    return error?.name === "AbortError";
}

function setQueryRunning(running) {
    elements.askButton.disabled = running;
    elements.askButton.textContent = running ? "Consultando…" : "Preguntar";
    elements.stopQueryButton.hidden = !running;
    elements.stopQueryButton.disabled = false;
}

function askForText({
    title,
    label,
    value = "",
    confirmText = "Guardar",
}) {
    if (!supportsDialog(elements.textDialog)) {
        return Promise.resolve(window.prompt(title, value));
    }

    return new Promise((resolve) => {
        elements.textDialogTitle.textContent = title;
        elements.textDialogLabel.textContent = label;
        elements.textDialogInput.value = value;
        elements.textDialogConfirm.textContent = confirmText;
        elements.textDialog.returnValue = "";

        const cleanup = () => {
            elements.textDialogInput.removeEventListener(
                "keydown",
                onKeyDown
            );
        };
        const close = () => {
            cleanup();
            resolve(
                elements.textDialog.returnValue === "confirm"
                    ? elements.textDialogInput.value
                    : null
            );
        };
        const onKeyDown = (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                elements.textDialog.close("confirm");
            }
        };

        elements.textDialog.addEventListener("close", close, {once: true});
        elements.textDialogInput.addEventListener("keydown", onKeyDown);
        elements.textDialog.showModal();
        elements.textDialogInput.focus();
        elements.textDialogInput.select();
    });
}

function askForConfirmation({
    title,
    message,
    confirmText = "Confirmar",
}) {
    if (!supportsDialog(elements.confirmDialog)) {
        return Promise.resolve(window.confirm(message));
    }

    return new Promise((resolve) => {
        elements.confirmDialogTitle.textContent = title;
        elements.confirmDialogMessage.textContent = message;
        elements.confirmDialogConfirm.textContent = confirmText;
        elements.confirmDialog.returnValue = "";

        const close = () => {
            resolve(elements.confirmDialog.returnValue === "confirm");
        };

        elements.confirmDialog.addEventListener("close", close, {
            once: true,
        });
        elements.confirmDialog.showModal();
        elements.confirmDialogConfirm.focus();
    });
}

function updateCounts(stats) {
    elements.documentCount.textContent = stats.documents ?? 0;
    elements.chunkCount.textContent = stats.chunks ?? 0;
}

function updateChatProfile(data) {
    for (const profile of data.chat_profiles) {
        const option = elements.chatProfile.querySelector(
            `option[value="${profile.id}"]`
        );
        if (option) {
            option.textContent = `${profile.label} · ${profile.model}`;
        }
    }

    const activeProfile = data.chat_profiles.find(
        (profile) => profile.id === data.active_chat_profile
    );
    if (activeProfile) {
        elements.customChatProfile.hidden = true;
        elements.chatProfile.value = activeProfile.id;
        applyContextCeiling(data.model_context);
        return;
    }

    const currentModel = data.ollama.chat_model.name;
    elements.customChatProfile.hidden = false;
    elements.customChatProfile.textContent =
        `Personalizado · ${currentModel}`;
    elements.chatProfile.value = "custom";
    applyContextCeiling(data.model_context);
}

function applyContextCeiling(modelContext) {
    // Un modelo no avisa si le pides más contexto del que se entrenó:
    // degrada la respuesta en silencio. Se marca el techo real (perfil,
    // Ollama o el fallback prudente) en la propia interfaz para que no
    // haga falta saberlo de memoria.
    const inputs = [
        elements.verifierContextTokens,
        elements.writerContextTokens,
    ];
    const ceiling = Number(modelContext?.model_context_window) || 32768;

    let excedido = false;
    for (const input of inputs) {
        if (!input) {
            continue;
        }
        input.max = String(ceiling);
        if (Number(input.value) > ceiling) {
            excedido = true;
        }
    }

    if (!elements.contextCeilingHint) {
        return;
    }
    if (!modelContext || !modelContext.verified) {
        elements.contextCeilingHint.textContent =
            `Techo de contexto sin verificar; se aplica un límite ` +
            `prudente de ${formatContextWindow(ceiling)} tokens.`;
        elements.contextCeilingHint.hidden = false;
        return;
    }
    if (excedido) {
        elements.contextCeilingHint.textContent =
            `El modelo activo admite ${formatContextWindow(ceiling)} ` +
            `tokens (${modelContext.source}). Los valores por encima se ` +
            "recortarán automáticamente al responder.";
        elements.contextCeilingHint.hidden = false;
        return;
    }
    elements.contextCeilingHint.hidden = true;
}

function formatTokenCount(value) {
    // Sin la palabra "tokens": la etiqueta <small> ya la muestra debajo
    // del campo, y repetirla aquí sólo añade ruido visual.
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) {
        return "";
    }
    return number % 1024 === 0 ? `${number / 1024}K` : String(number);
}

function updateContextValueReadout(input, output) {
    if (!input || !output) {
        return;
    }
    output.textContent = formatTokenCount(input.value);
}

function documentStatus(metadata) {
    const raw = String(metadata?.estado ?? metadata?.status ?? "")
        .trim()
        .toLocaleLowerCase("es");
    const known = {
        vigente: {label: "Vigente", className: "vigente"},
        borrador: {label: "Borrador", className: "borrador"},
        derogado: {label: "Derogado", className: "derogado"},
    };
    return known[raw] ?? {
        label: raw || "Sin estado",
        className: "sin-estado",
    };
}

function safeDomId(value) {
    return String(value).replace(/[^a-zA-Z0-9_-]/g, "-");
}

function setChatLayout(hasMessages, title = "") {
    elements.workspace.classList.toggle("has-messages", hasMessages);
    elements.welcome.classList.toggle("hidden", hasMessages);
    elements.chatThread.classList.toggle("hidden", !hasMessages);
    elements.conversationHeader.classList.toggle("hidden", !hasMessages);
    elements.conversationTitle.textContent = title || "Nueva conversación";
}

function preferredMemoryEnabled() {
    return window.localStorage.getItem(MEMORY_PREFERENCE_KEY) !== "false";
}

function updateMemoryStatus(memory = {}) {
    const enabled = memory.enabled ?? elements.useMemory.checked;
    if (!enabled) {
        elements.memoryStatus.textContent =
            "Desactivada: las preguntas se procesan de forma independiente.";
        return;
    }
    if (memory.warning || memory.last_error) {
        elements.memoryStatus.textContent =
            "Activa · el último resumen no pudo actualizarse; " +
            "se mantienen los turnos guardados.";
        return;
    }
    const summarized = Number(memory.summarized_turns || 0);
    const pending = Number(memory.pending_turns || 0);
    if (memory.summary_updated) {
        elements.memoryStatus.textContent =
            `Activa · resumen actualizado (${summarized} turnos) · ` +
            `${pending} pendientes.`;
        return;
    }
    if (memory.has_summary) {
        elements.memoryStatus.textContent =
            `Activa · ${summarized} turnos resumidos · ` +
            `${pending} pendientes.`;
        return;
    }
    if (pending > 0) {
        elements.memoryStatus.textContent =
            `Activa · ${pending}/${memorySummaryInterval} turnos para ` +
            "crear el primer resumen.";
        return;
    }
    elements.memoryStatus.textContent =
        `Activa · resumen cada ${memorySummaryInterval} turnos y ` +
        `hasta ${memoryRecentTurns} turnos recientes.`;
}

function scrollToLatestTurn() {
    requestAnimationFrame(() => {
        elements.chatThread.lastElementChild?.scrollIntoView({
            behavior: "smooth",
            block: "end",
        });
    });
}

function focusSource(turnKey, referenceNumber) {
    const card = document.getElementById(
        `source-${safeDomId(turnKey)}-${referenceNumber}`
    );
    if (!card) {
        return;
    }
    const sourceGroup = card.closest(".turn-sources");
    if (sourceGroup) {
        sourceGroup.open = true;
    }
    card.open = true;
    card.classList.remove("source-highlight");
    void card.offsetWidth;
    card.classList.add("source-highlight");
    card.scrollIntoView({behavior: "smooth", block: "center"});
}

function renderAnswer(container, answer, sourceCount, turnKey) {
    container.replaceChildren();
    const citationPattern = /\[(\d+)\]/g;
    let cursor = 0;

    for (const match of answer.matchAll(citationPattern)) {
        container.append(
            document.createTextNode(answer.slice(cursor, match.index))
        );
        const referenceNumber = Number(match[1]);
        if (referenceNumber >= 1 && referenceNumber <= sourceCount) {
            const citation = document.createElement("button");
            citation.type = "button";
            citation.className = "citation";
            citation.textContent = match[0];
            citation.title = `Ver fuente ${referenceNumber}`;
            citation.setAttribute(
                "aria-label",
                `Ver fuente ${referenceNumber}`
            );
            citation.addEventListener(
                "click",
                () => focusSource(turnKey, referenceNumber)
            );
            container.append(citation);
        } else {
            container.append(document.createTextNode(match[0]));
        }
        cursor = match.index + match[0].length;
    }
    container.append(document.createTextNode(answer.slice(cursor)));
}

async function copyTextToClipboard(text) {
    if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return;
    }

    const copyArea = document.createElement("textarea");
    copyArea.value = text;
    copyArea.setAttribute("readonly", "");
    copyArea.className = "clipboard-fallback";
    document.body.append(copyArea);
    copyArea.select();
    const copied = document.execCommand("copy");
    copyArea.remove();
    if (!copied) {
        throw new Error("No se pudo copiar la respuesta.");
    }
}

function createCopyAnswerButton(getText) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-answer-button";
    button.textContent = "Copiar";
    button.title = "Copiar respuesta";
    button.setAttribute("aria-label", "Copiar respuesta al portapapeles");

    button.addEventListener("click", async () => {
        const text = getText().trim();
        if (!text) {
            return;
        }
        button.disabled = true;
        try {
            await copyTextToClipboard(text);
            button.textContent = "Copiado";
        } catch (error) {
            button.textContent = "No copiado";
            console.error(error);
        } finally {
            window.setTimeout(() => {
                button.disabled = false;
                button.textContent = "Copiar";
            }, 1600);
        }
    });

    return button;
}

function createAssistantHeader(label, actions = []) {
    const header = document.createElement("div");
    header.className = "assistant-header";
    header.append(label, ...actions);
    return header;
}

const VIGENCIA_LABELS = {
    caducada: {label: "Derogado", className: "vigencia-caducada"},
    futura: {label: "Aún no vigente", className: "vigencia-futura"},
};

function vigenciaBadge(vigencia) {
    // Sólo se avisa de lo que puede inducir a error. Un documento
    // vigente no necesita distintivo: es lo esperado.
    if (!vigencia || !vigencia.estado_temporal) {
        return null;
    }
    const config = VIGENCIA_LABELS[vigencia.estado_temporal];
    if (!config) {
        return null;
    }
    const badge = document.createElement("span");
    badge.className = `vigencia-badge ${config.className}`;
    badge.textContent = config.label;
    const detalle = [];
    if (vigencia.valid_from) {
        detalle.push(`en vigor desde ${vigencia.valid_from}`);
    }
    if (vigencia.valid_until) {
        detalle.push(`derogado el ${vigencia.valid_until}`);
    }
    if (vigencia.derogado_por_callout) {
        detalle.push("el propio documento se declara derogado");
    }
    badge.title = detalle.join(" · ") || config.label;
    return badge;
}

function createSourceCard(source, index, turnKey) {
    const card = document.createElement("details");
    card.className = "source-card";
    card.id = `source-${safeDomId(turnKey)}-${index}`;

    const summary = document.createElement("summary");
    summary.className = "source-summary";

    const reference = document.createElement("div");
    reference.className = "source-reference";
    reference.textContent = source.reference;

    const overview = document.createElement("div");
    overview.className = "source-overview";

    const titleRow = document.createElement("div");
    titleRow.className = "source-title-row";
    const title = document.createElement("h4");
    title.className = "source-title";
    title.textContent = source.title;

    const statusData = documentStatus(source.metadata);
    const status = document.createElement("span");
    status.className = `document-status ${statusData.className}`;
    status.textContent = statusData.label;
    titleRow.append(title, status);

    const vigencia = vigenciaBadge(source.vigencia);
    if (vigencia) {
        titleRow.append(vigencia);
    }
    for (const tag of (source.tags || []).slice(0, 4)) {
        const chip = document.createElement("span");
        chip.className = "source-tag";
        chip.textContent = `#${tag}`;
        titleRow.append(chip);
    }

    const detail = document.createElement("p");
    detail.className = "source-detail";
    detail.textContent =
        `${source.path} · ${source.heading} · ${source.reason}`;
    overview.append(titleRow, detail);

    const score = document.createElement("div");
    score.className = "source-score";
    score.textContent = Number(source.score).toFixed(4);
    const scoreParts = ["Puntuación de recuperación"];
    if (source.semantic_score != null) {
        scoreParts.push(
            `similitud semántica ${Number(source.semantic_score).toFixed(4)}`
        );
    }
    if (source.lexical_score != null) {
        scoreParts.push(
            `BM25 textual ${Number(source.lexical_score).toFixed(4)}`
        );
    }
    score.title = scoreParts.join(" · ");

    const chevron = document.createElement("span");
    chevron.className = "source-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "›";
    summary.append(reference, overview, score, chevron);

    const fragment = document.createElement("div");
    fragment.className = "source-fragment";
    const fragmentLabel = document.createElement("span");
    fragmentLabel.className = "source-fragment-label";
    fragmentLabel.textContent = source.context_expanded
        ? "Contexto padre"
        : "Fragmento recuperado";
    const fragmentText = document.createElement("p");
    fragmentText.textContent = source.content;
    fragment.append(fragmentLabel, fragmentText);
    if (
        source.context_expanded &&
        source.matched_content &&
        source.matched_content !== source.content
    ) {
        const matchLabel = document.createElement("span");
        matchLabel.className = "source-fragment-label";
        const matchCount = Array.isArray(source.matched_chunk_ids)
            ? source.matched_chunk_ids.length
            : 1;
        matchLabel.textContent =
            matchCount > 1
                ? `Coincidencia encontrada (mejor de ${matchCount} fragmentos de esta sección)`
                : "Coincidencia encontrada";
        const matchText = document.createElement("p");
        matchText.textContent = source.matched_content;
        fragment.append(matchLabel, matchText);
    }
    card.append(summary, fragment);
    return card;
}

function agentTraceTooltip(roleMetrics) {
    const parts = [];
    if (roleMetrics.model) {
        parts.push(`Modelo ${roleMetrics.model}`);
    }
    if (roleMetrics.effective_context_window) {
        parts.push(
            `${roleMetrics.total_tokens ?? "?"}/` +
            `${roleMetrics.effective_context_window} tokens`
        );
    }
    parts.push(roleMetrics.estimated ? "estimado" : "medido");
    if (roleMetrics.incomplete_context_data) {
        parts.push("dato histórico incompleto");
    }
    return parts.join(" · ");
}

function createAgentTrace(turn) {
    const agentMetrics = turn?.agent_metrics || turn?.agents;
    if (!agentMetrics || typeof agentMetrics !== "object") {
        return null;
    }
    const trace = document.createElement("div");
    trace.className = "agent-trace";
    const verifier = agentMetrics.verifier || {};
    const writer = agentMetrics.writer || {};

    const verifierItem = document.createElement("span");
    if (verifier.status === "completed" || verifier.status === "degraded") {
        verifierItem.textContent =
            `Verificador ${formatContextPercent(
                verifier.usage_percent
            )}`;
        verifierItem.title = agentTraceTooltip(verifier);
    } else if (verifier.status === "error") {
        verifierItem.textContent = "Verificador no disponible";
        verifierItem.classList.add("agent-error");
    } else if (verifier.status === "disabled") {
        verifierItem.textContent = "Verificación desactivada";
    } else {
        verifierItem.textContent = "Sin verificación";
    }
    trace.append(verifierItem);

    if (writer.status === "completed" || writer.status === "degraded") {
        const writerItem = document.createElement("span");
        writerItem.textContent =
            `Redactor ${formatContextPercent(writer.usage_percent)}`;
        writerItem.title = agentTraceTooltip(writer);
        trace.append(writerItem);
    }

    const detailsButton = document.createElement("button");
    detailsButton.type = "button";
    detailsButton.className = "agent-trace-details";
    detailsButton.textContent = "Contexto";
    detailsButton.title =
        "Ver el modelo y la ventana de contexto usados en este turno";
    detailsButton.addEventListener("click", () => {
        openTurnInspector(turn.chat_model, agentMetrics);
    });
    trace.append(detailsButton);

    return trace;
}

// ---------------------------------------------------------------------
// Streaming de respuestas (SSE sobre fetch)
// ---------------------------------------------------------------------
let streamingSupported = true;

const STAGE_LABELS = {
    verifying: "Verificando las fuentes…",
    writing: "Redactando la respuesta…",
};

async function* readServerEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const {done, value} = await reader.read();
        if (done) {
            break;
        }
        buffer += decoder.decode(value, {stream: true});

        let separator = buffer.indexOf("\n\n");
        while (separator !== -1) {
            const block = buffer.slice(0, separator);
            buffer = buffer.slice(separator + 2);
            let name = "message";
            const dataLines = [];
            for (const line of block.split("\n")) {
                if (line.startsWith("event: ")) {
                    name = line.slice(7).trim();
                } else if (line.startsWith("data: ")) {
                    dataLines.push(line.slice(6));
                }
            }
            if (dataLines.length) {
                try {
                    yield {name, data: JSON.parse(dataLines.join("\n"))};
                } catch (error) {
                    console.error("Evento SSE ilegible", error);
                }
            }
            separator = buffer.indexOf("\n\n");
        }
    }
}

function createStreamingTurn(question) {
    const turnKey = `stream-${++pendingTurnNumber}`;
    const wrapper = document.createElement("article");
    wrapper.className = "chat-turn";
    wrapper.dataset.turnId = String(turnKey);

    const userRow = document.createElement("div");
    userRow.className = "message-row user-row";
    const userBubble = document.createElement("div");
    userBubble.className = "message-bubble user-bubble";
    userBubble.textContent = question;
    userRow.append(userBubble);

    const assistantRow = document.createElement("div");
    assistantRow.className = "message-row assistant-row";
    const assistantBody = document.createElement("div");
    assistantBody.className = "assistant-body";

    const assistantLabel = document.createElement("div");
    assistantLabel.className = "assistant-label";
    assistantLabel.textContent = "Obsidian RAG";

    const stage = document.createElement("div");
    stage.className = "stream-stage";
    stage.textContent = "Buscando en la documentación…";

    const answer = document.createElement("div");
    answer.className = "answer-card streaming-answer";
    // La respuesta se anuncia entera al terminar, no token a token: el
    // progreso real lo lleva #query-status, que sí es una región viva.
    answer.setAttribute("aria-busy", "true");

    const sourcesHolder = document.createElement("div");
    sourcesHolder.className = "stream-sources";

    assistantBody.append(
        createAssistantHeader(assistantLabel),
        stage,
        answer,
        sourcesHolder
    );
    assistantRow.append(assistantBody);
    wrapper.append(userRow, assistantRow);

    return {
        wrapper,
        turnKey,
        assistantBody,
        assistantLabel,
        stage,
        answer,
        sourcesHolder,
        text: "",
        sources: [],
    };
}

function renderSourcesInto(holder, sources, turnKey) {
    holder.replaceChildren();
    if (!sources.length) {
        return;
    }
    const group = document.createElement("details");
    group.className = "turn-sources";
    const summary = document.createElement("summary");
    summary.textContent = `${sources.length} fuentes documentales`;
    const grid = document.createElement("div");
    grid.className = "sources-grid";
    sources.forEach((source, index) => {
        grid.append(createSourceCard(source, index + 1, turnKey));
    });
    group.append(summary, grid);
    holder.append(group);
}

async function streamQuery(question, body, signal) {
    const turn = createStreamingTurn(question);
    elements.chatThread.append(turn.wrapper);
    setChatLayout(true, activeConversationTitle || "Nueva conversación");
    scrollToLatestTurn();

    let response;
    try {
        response = await fetch("/api/query/stream", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
            signal,
        });
    } catch (error) {
        turn.wrapper.remove();
        throw error;
    }

    if (!response.ok || !response.body) {
        turn.wrapper.remove();
        if (response.status === 409 || response.status === 404) {
            // El servidor no ofrece streaming: se usa la vía clásica.
            streamingSupported = false;
            return null;
        }
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "No se pudo iniciar la consulta.");
    }

    let finalPayload = null;
    let failure = null;
    let atBottom = true;

    try {
    for await (const event of readServerEvents(response)) {
        if (event.name === "retrieval") {
            turn.sources = event.data.sources || [];
            renderSourcesInto(
                turn.sourcesHolder,
                turn.sources,
                turn.turnKey
            );
            turn.stage.textContent = turn.sources.length
                ? `${turn.sources.length} fuentes recuperadas`
                : "Sin fuentes suficientes";
            elements.queryStatus.textContent = turn.stage.textContent;
            scrollToLatestTurn();
        } else if (event.name === "stage") {
            turn.stage.textContent =
                STAGE_LABELS[event.data.stage] || turn.stage.textContent;
            elements.queryStatus.textContent = turn.stage.textContent;
        } else if (event.name === "delta") {
            turn.text += event.data.text;
            turn.answer.textContent = turn.text;
            if (atBottom) {
                scrollToLatestTurn();
            }
            atBottom = isThreadNearBottom();
        } else if (event.name === "done") {
            finalPayload = event.data;
        } else if (event.name === "error") {
            failure = event.data.detail || "La consulta no se completó.";
        }
    }

    } catch (error) {
        turn.wrapper.remove();
        throw error;
    }

    if (failure) {
        turn.wrapper.remove();
        throw new Error(failure);
    }
    if (!finalPayload) {
        turn.wrapper.remove();
        throw new Error("La conexión se interrumpió antes de terminar.");
    }

    // Sustituye el turno en vivo por el definitivo: citas pulsables,
    // traza de agentes y misma estructura que el historial guardado.
    const completed = createTurnElement({
        id: finalPayload.turn_id,
        question,
        answer: finalPayload.answer,
        sources: turn.sources,
        chat_model: finalPayload.chat_model,
        agents: finalPayload.agents,
    });
    turn.wrapper.replaceWith(completed);
    return finalPayload;
}

function isThreadNearBottom() {
    const thread = elements.chatThread;
    if (!thread) {
        return true;
    }
    const distance =
        thread.scrollHeight - thread.scrollTop - thread.clientHeight;
    return distance < 120;
}

function createTurnElement(turn, options = {}) {
    const turnKey = turn.id ?? `pending-${++pendingTurnNumber}`;
    const wrapper = document.createElement("article");
    wrapper.className = "chat-turn";
    wrapper.dataset.turnId = String(turnKey);

    const userRow = document.createElement("div");
    userRow.className = "message-row user-row";
    const userBubble = document.createElement("div");
    userBubble.className = "message-bubble user-bubble";
    userBubble.textContent = turn.question;
    userRow.append(userBubble);

    const assistantRow = document.createElement("div");
    assistantRow.className = "message-row assistant-row";
    const assistantBody = document.createElement("div");
    assistantBody.className = "assistant-body";
    const assistantLabel = document.createElement("div");
    assistantLabel.className = "assistant-label";
    assistantLabel.textContent = turn.chat_model
        ? `Obsidian RAG · ${turn.chat_model}`
        : "Obsidian RAG";
    const answer = document.createElement("div");
    answer.className = "answer-card";

    if (options.pending) {
        answer.classList.add("loading-answer");
        answer.textContent = "Buscando en la documentación…";
    } else if (options.error) {
        answer.classList.add("error-answer");
        answer.textContent = options.error;
    } else {
        const sources = Array.isArray(turn.sources) ? turn.sources : [];
        const answerText = turn.answer || "";
        renderAnswer(answer, answerText, sources.length, turnKey);
        assistantBody.append(
            createAssistantHeader(assistantLabel, [
                createCopyAnswerButton(() => answerText),
            ])
        );
        const agentTrace = createAgentTrace(turn);
        if (agentTrace) {
            assistantBody.append(agentTrace);
        }
        assistantBody.append(answer);
        if (sources.length) {
            const sourceGroup = document.createElement("details");
            sourceGroup.className = "turn-sources";
            const sourceSummary = document.createElement("summary");
            sourceSummary.textContent =
                `${sources.length} fuentes documentales`;
            const sourceGrid = document.createElement("div");
            sourceGrid.className = "sources-grid";
            sources.forEach((source, index) => {
                sourceGrid.append(
                    createSourceCard(source, index + 1, turnKey)
                );
            });
            sourceGroup.append(sourceSummary, sourceGrid);
            assistantBody.append(sourceGroup);
            assistantRow.append(assistantBody);
            wrapper.append(userRow, assistantRow);
            return wrapper;
        }
    }

    if (!assistantLabel.parentElement) {
        assistantBody.append(createAssistantHeader(assistantLabel), answer);
    }
    assistantRow.append(assistantBody);
    wrapper.append(userRow, assistantRow);
    return wrapper;
}

function renderConversation(conversation) {
    activeConversationId = conversation.id;
    activeConversationTitle = conversation.title;
    const memory = conversation.memory || {
        enabled: conversation.memory_enabled !== false,
    };
    elements.useMemory.checked = memory.enabled !== false;
    window.localStorage.setItem(
        MEMORY_PREFERENCE_KEY,
        String(elements.useMemory.checked)
    );
    updateMemoryStatus(memory);
    elements.chatThread.replaceChildren();
    for (const turn of conversation.turns) {
        elements.chatThread.append(createTurnElement(turn));
    }
    setChatLayout(conversation.turns.length > 0, conversation.title);
    markActiveConversation();
    scrollToLatestTurn();
    const lastTurn = conversation.turns[conversation.turns.length - 1];
    latestTurnMaxUsagePercent = maxUsagePercentFromAgents(
        lastTurn?.agent_metrics
    );
    refreshContextIndicatorHeader();
}

function startNewConversation() {
    setMobileTab("chat");
    activeConversationId = null;
    activeConversationTitle = "";
    elements.chatThread.replaceChildren();
    elements.queryStatus.textContent = "";
    elements.useMemory.checked = preferredMemoryEnabled();
    updateMemoryStatus({enabled: elements.useMemory.checked});
    setChatLayout(false);
    markActiveConversation();
    latestTurnMaxUsagePercent = null;
    refreshContextIndicatorHeader();
    elements.question.focus();
}

function markActiveConversation() {
    for (const item of elements.conversationList.querySelectorAll(
        ".conversation-item"
    )) {
        item.classList.toggle(
            "active",
            item.dataset.conversationId === activeConversationId
        );
    }
}

function markActiveProject() {
    for (const item of elements.projectList.querySelectorAll(
        ".project-item"
    )) {
        item.classList.toggle(
            "active",
            item.dataset.projectId === activeProjectId
        );
    }
}

function updateDeleteProjectButton() {
    const canDelete =
        Boolean(activeProjectId) && activeProjectId !== defaultProjectId;
    elements.deleteProjectButton.disabled = !canDelete;
    elements.deleteProjectButton.title = canDelete
        ? `Eliminar proyecto ${activeProjectName || "seleccionado"}`
        : "El proyecto General no se puede eliminar";
}

function updateActiveProjectIndicators(projects = []) {
    const activeProject = projects.find(
        (project) => project.id === activeProjectId
    );
    if (activeProject) {
        activeProjectName = activeProject.name;
    }
    const label = activeProjectName || "Sin proyecto";
    elements.workspaceActiveProject.textContent = label;
    elements.workspaceActiveProject.title = label;
    updateDeleteProjectButton();
}

function formatContextPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(1)}%` : "—";
}

function contextLevel(percent) {
    // Umbrales del inspector de contexto: seguro <60%, advertencia
    // 60-85%, peligro >85%. No es sólo color: el texto y el aria-label
    // que acompañan a estas barras llevan la cifra exacta.
    if (!Number.isFinite(percent)) {
        return "empty";
    }
    if (percent > 85) {
        return "danger";
    }
    if (percent >= 60) {
        return "warning";
    }
    return "safe";
}

function formatContextWindow(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) {
        return "—";
    }
    if (number >= 1024) {
        const kilo = number / 1024;
        return Number.isInteger(kilo) ? `${kilo}K` : `${kilo.toFixed(1)}K`;
    }
    return String(number);
}

const AGENT_STATUS_LABELS = {
    completed: "Activado",
    degraded: "Activado (degradado)",
    disabled: "Desactivado",
    pending: "Pendiente",
    skipped: "Omitido",
    skipped_no_evidence: "Omitido: sin evidencia recuperada",
    skipped_insufficient_evidence: "Omitido: evidencia insuficiente",
    error: "Error",
};

function maxUsagePercentFromAgents(agents) {
    if (!agents || typeof agents !== "object") {
        return null;
    }
    const percentages = [];
    for (const role of ["verifier", "writer"]) {
        const roleMetrics = agents[role];
        const status = roleMetrics?.status;
        if (status === "completed" || status === "degraded") {
            const value = Number(roleMetrics.usage_percent);
            if (Number.isFinite(value)) {
                percentages.push(value);
            }
        }
    }
    return percentages.length ? Math.max(...percentages) : null;
}

// ---------------------------------------------------------------------
// Inspector de contexto
// ---------------------------------------------------------------------
function refreshContextIndicatorHeader() {
    if (!elements.contextIndicator) {
        return;
    }
    const modelContext = lastStatusData?.model_context;
    if (!modelContext) {
        elements.contextIndicatorPercent.textContent = "—";
        elements.contextIndicatorModel.textContent = "Cargando…";
        elements.contextIndicatorWindow.textContent = "—/—";
        return;
    }
    const project = loadedProjects.find(
        (candidate) => candidate.id === activeProjectId
    );
    const configuredWriter =
        Number(project?.agent_settings?.writer_context_tokens) || 16384;
    const effective = Math.min(
        configuredWriter,
        modelContext.model_context_window
    );
    const percent =
        latestTurnMaxUsagePercent ?? project?.context_usage?.latest_percent;

    elements.contextIndicatorPercent.textContent =
        percent == null ? "—" : formatContextPercent(percent);
    elements.contextIndicatorModel.textContent = modelContext.model;
    elements.contextIndicatorWindow.textContent =
        `${formatContextWindow(effective)}/` +
        `${formatContextWindow(modelContext.model_context_window)}`;
    elements.contextIndicator.dataset.level = contextLevel(
        Number(percent)
    );
    elements.contextIndicator.title = modelContext.verified
        ? `Fuente de la capacidad: ${modelContext.source}. Pulsa para ` +
          "abrir el inspector de contexto."
        : `Capacidad sin verificar (${modelContext.source}); se usa un ` +
          "límite prudente. Pulsa para abrir el inspector de contexto.";
}

function renderContextRolePanel(container, metrics) {
    if (!container) {
        return;
    }
    container.replaceChildren();
    if (!metrics || typeof metrics !== "object" || !metrics.status) {
        const empty = document.createElement("p");
        empty.className = "context-inspector-empty";
        empty.textContent = "Todavía no hay datos para este agente.";
        container.append(empty);
        return;
    }

    const rows = [
        ["Estado", AGENT_STATUS_LABELS[metrics.status] || metrics.status],
        ["Modelo usado", metrics.model || "—"],
        [
            "Ventana configurada",
            formatContextWindow(metrics.configured_context_window),
        ],
        [
            "Ventana efectiva",
            formatContextWindow(metrics.effective_context_window),
        ],
    ];
    const measuring =
        metrics.status === "completed" || metrics.status === "degraded";
    if (measuring) {
        rows.push(
            ["Entrada", `${metrics.prompt_tokens ?? "—"} tokens`],
            ["Salida", `${metrics.completion_tokens ?? "—"} tokens`],
            ["Total", `${metrics.total_tokens ?? "—"} tokens`],
            [
                "Margen disponible",
                `${metrics.remaining_tokens ?? "—"} tokens`,
            ]
        );
    }

    const dl = document.createElement("dl");
    dl.className = "context-inspector-grid";
    for (const [label, value] of rows) {
        const row = document.createElement("div");
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = value;
        row.append(dt, dd);
        dl.append(row);
    }
    container.append(dl);

    if (measuring) {
        const percent = Number(metrics.usage_percent);
        const meterWrap = document.createElement("div");
        meterWrap.className = "context-meter";
        meterWrap.setAttribute("role", "img");
        meterWrap.setAttribute(
            "aria-label",
            `Uso de contexto: ${formatContextPercent(percent)}`
        );
        const meterFill = document.createElement("span");
        meterFill.dataset.level = contextLevel(percent);
        meterFill.style.width = Number.isFinite(percent)
            ? `${Math.min(100, Math.max(0, percent))}%`
            : "0%";
        meterWrap.append(meterFill);
        container.append(meterWrap);

        const measurementNote = document.createElement("p");
        measurementNote.className = "context-inspector-note";
        measurementNote.textContent = metrics.estimated
            ? "Estimado: Ollama no informó contadores exactos en esta " +
              "inferencia."
            : "Medido: contadores reales informados por Ollama.";
        container.append(measurementNote);
    }

    if (metrics.context_warning) {
        const warning = document.createElement("p");
        warning.className = "context-inspector-warning";
        warning.textContent = metrics.context_warning;
        container.append(warning);
    }

    if (metrics.incomplete_context_data) {
        const incomplete = document.createElement("p");
        incomplete.className = "context-inspector-warning";
        incomplete.textContent =
            "Dato histórico incompleto: este turno se guardó antes de " +
            "registrar todos estos campos.";
        container.append(incomplete);
    }
}

function renderContextInspectorModel(modelContext) {
    if (!modelContext) {
        elements.ciModelName.textContent = "—";
        elements.ciModelProfile.textContent = "—";
        elements.ciModelCapacity.textContent = "—";
        elements.ciModelSource.textContent = "—";
        elements.ciModelVerified.textContent = "—";
        elements.ciModelWarning.hidden = true;
        return;
    }
    elements.ciModelName.textContent = modelContext.model || "—";
    elements.ciModelProfile.textContent = modelContext.profile_id || "Ninguno";
    elements.ciModelCapacity.textContent = formatContextWindow(
        modelContext.model_context_window
    );
    const sourceLabels = {
        profile: "Perfil conocido",
        ollama: "Metadatos de Ollama",
        "profile+ollama": "Perfil + Ollama",
        fallback: "Límite prudente (sin datos)",
    };
    elements.ciModelSource.textContent =
        sourceLabels[modelContext.source] || modelContext.source || "—";
    elements.ciModelVerified.textContent = modelContext.verified
        ? "Sí"
        : "No";
    if (modelContext.warning) {
        elements.ciModelWarning.textContent = modelContext.warning;
        elements.ciModelWarning.hidden = false;
    } else {
        elements.ciModelWarning.hidden = true;
    }
}

function renderContextInspectorConversation(overview) {
    renderContextInspectorModel(overview.current_model);

    elements.ciConversationEmpty.hidden = true;
    elements.ciConversationGrid.hidden = false;
    const history = overview.history || {};
    elements.ciTotalTurns.textContent = String(history.total_turns ?? 0);
    elements.ciStoredTokens.textContent =
        `~${history.stored_estimated_tokens ?? 0} tokens (estimado)`;
    elements.ciRecentTurns.textContent = String(
        history.recent_turns_included ?? 0
    );
    const recentParts = [`~${history.recent_estimated_tokens ?? 0} tokens`];
    if (history.summary_included) {
        recentParts.push("+ resumen");
    }
    elements.ciRecentTokens.textContent =
        `${recentParts.join(" ")} (estimado)`;
    elements.ciSummaryTokens.textContent = history.summary_included
        ? `~${history.summary_estimated_tokens ?? 0} tokens (estimado)`
        : "Sin resumen todavía";
    elements.ciProjectMemoryTokens.textContent =
        history.project_memory_estimated_tokens
            ? `~${history.project_memory_estimated_tokens} tokens (estimado)`
            : "Sin memoria de proyecto";

    const nextEstimate = overview.next_inference_estimate || {};
    elements.ciNextEstimate.textContent =
        `~${nextEstimate.estimated_prompt_tokens ?? "—"} tokens de ` +
        `${formatContextWindow(nextEstimate.effective_context_window)} ` +
        `(${formatContextPercent(nextEstimate.estimated_usage_percent)}, ` +
        "estimado)";

    const lastInference = overview.last_inference || {};
    renderContextRolePanel(elements.ciVerifier, lastInference.verifier);
    renderContextRolePanel(elements.ciWriter, lastInference.writer);

    const warnings = Array.isArray(overview.warnings) ? overview.warnings : [];
    if (warnings.length) {
        elements.ciWarnings.textContent = warnings.join(" ");
        elements.ciWarnings.hidden = false;
    } else {
        elements.ciWarnings.hidden = true;
    }
}

async function refreshContextInspector() {
    renderContextInspectorModel(lastStatusData?.model_context);
    if (!activeConversationId) {
        elements.ciConversationEmpty.hidden = false;
        elements.ciConversationGrid.hidden = true;
        renderContextRolePanel(elements.ciVerifier, null);
        renderContextRolePanel(elements.ciWriter, null);
        elements.ciWarnings.hidden = true;
        return;
    }
    try {
        const overview = await apiRequest(
            `/api/conversations/${encodeURIComponent(
                activeConversationId
            )}/context`
        );
        renderContextInspectorConversation(overview);
    } catch (error) {
        elements.ciConversationEmpty.hidden = false;
        elements.ciConversationEmpty.textContent = error.message;
        elements.ciConversationGrid.hidden = true;
    }
}

function openContextInspector() {
    if (!supportsDialog(elements.contextInspectorDialog)) {
        elements.systemMessage.textContent =
            "El inspector de contexto requiere un navegador con soporte " +
            "para <dialog>.";
        return;
    }
    elements.contextInspectorDialog.showModal();
    refreshContextInspector();
}

function openTurnInspector(chatModel, agentMetrics) {
    if (!supportsDialog(elements.turnInspectorDialog)) {
        elements.queryStatus.textContent =
            "El detalle de contexto requiere un navegador con soporte " +
            "para <dialog>.";
        return;
    }
    elements.turnInspectorModel.textContent = chatModel
        ? `Modelo utilizado en este turno: ${chatModel}`
        : "Modelo utilizado en este turno: dato histórico incompleto.";
    renderContextRolePanel(
        elements.turnInspectorVerifier,
        agentMetrics?.verifier
    );
    renderContextRolePanel(
        elements.turnInspectorWriter,
        agentMetrics?.writer
    );
    elements.turnInspectorDialog.showModal();
}

elements.contextIndicator?.addEventListener("click", openContextInspector);

const AGENT_USAGE_ELEMENTS = {
    verifier: () => [
        elements.verifierContextUsage,
        elements.verifierContextBar,
    ],
    writer: () => [elements.writerContextUsage, elements.writerContextBar],
};

function renderAgentUsage(role, usage) {
    const latest = usage?.latest;
    const percent = latest ? Number(latest.usage_percent) : NaN;
    const [usageElement, barElement] = AGENT_USAGE_ELEMENTS[role]();
    const usedPercent = Number.isFinite(percent)
        ? Math.min(100, Math.max(0, percent))
        : NaN;
    const remainingPercent = Number.isFinite(usedPercent)
        ? 100 - usedPercent
        : NaN;
    usageElement.textContent = Number.isFinite(percent)
        ? `${formatContextPercent(remainingPercent)} disponible`
        : "Sin datos";
    usageElement.dataset.level = contextLevel(percent);
    barElement.style.width = Number.isFinite(usedPercent)
        ? `${usedPercent}%`
        : "0%";
    barElement.dataset.level = contextLevel(percent);
    usageElement.title = usage?.samples
        ? `Usado ${formatContextPercent(percent)} · ` +
          `${latest.total_tokens}/${latest.context_window_tokens} tokens. ` +
          `Máximo ${formatContextPercent(usage.maximum_percent)} · ` +
          `media ${formatContextPercent(usage.average_percent)}.`
        : "Todavía no hay consultas registradas para este agente.";
}

function setAgentTogglesEnabled(enabled) {
    // Viven en el panel de Opciones, fuera del bloque que se oculta al
    // no haber proyecto: se deshabilitan en lugar de desaparecer, para
    // que el panel no cambie de tamaño al cambiar de proyecto.
    elements.projectVerificationEnabled.disabled = !enabled;
    elements.projectMemoryEnabled.disabled = !enabled;
}

function renderProjectAgentSettings(project) {
    if (!project) {
        elements.projectAgentSettings.hidden = true;
        setAgentTogglesEnabled(false);
        return;
    }
    setAgentTogglesEnabled(true);
    const settings = project.agent_settings || {};
    const usage = project.context_usage || {};
    const memory = project.memory || {};
    elements.projectAgentSettings.hidden = false;
    elements.projectVerificationEnabled.checked =
        settings.verification_enabled !== false;
    elements.projectMemoryEnabled.checked =
        settings.project_memory_enabled !== false;
    elements.verifierContextTokens.value =
        settings.verifier_context_tokens ?? 8192;
    elements.writerContextTokens.value =
        settings.writer_context_tokens ?? 16384;
    elements.verifierContextTokens.disabled =
        !elements.projectVerificationEnabled.checked;
    updateContextValueReadout(
        elements.verifierContextTokens,
        elements.verifierContextValue
    );
    updateContextValueReadout(
        elements.writerContextTokens,
        elements.writerContextValue
    );
    elements.projectContextOverall.textContent =
        usage.latest_percent == null
            ? "Sin uso registrado"
            : `Última consulta: ${formatContextPercent(
                usage.latest_percent
            )}`;
    renderAgentUsage("verifier", usage.verifier);
    renderAgentUsage("writer", usage.writer);
    if (!elements.projectMemoryEnabled.checked) {
        elements.projectMemorySummaryStatus.textContent =
            "Memoria compartida desactivada; el resumen se conserva.";
    } else if (memory.last_error) {
        elements.projectMemorySummaryStatus.textContent =
            "La última actualización de memoria falló.";
    } else if (memory.has_summary) {
        elements.projectMemorySummaryStatus.textContent =
            `${memory.summarized_turns || 0} turnos resumidos · ` +
            `${memory.pending_turns || 0} pendientes.`;
    } else {
        elements.projectMemorySummaryStatus.textContent =
            `${memory.pending_turns || 0} turnos pendientes para el ` +
            "primer resumen compartido.";
    }
}

async function selectProject(
    projectId,
    projectName,
    {selectLatest = true} = {}
) {
    if (!projectId) {
        return;
    }
    activeProjectId = projectId;
    activeProjectName = projectName || "";
    window.localStorage.setItem(ACTIVE_PROJECT_KEY, projectId);
    markActiveProject();
    updateActiveProjectIndicators();
    renderProjectAgentSettings(
        loadedProjects.find((project) => project.id === projectId)
    );
    startNewConversation();
    await loadConversations({selectLatest});
}

function createProjectListItem(project) {
    const item = document.createElement("div");
    item.className = "project-item";
    item.dataset.projectId = project.id;

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "project-open";
    openButton.title = project.name;
    const name = document.createElement("span");
    name.textContent = project.name;
    const count = document.createElement("small");
    count.textContent =
        `${project.conversation_count} ` +
        (project.conversation_count === 1
            ? "conversación"
            : "conversaciones");
    const context = document.createElement("small");
    context.className = "project-context-usage";
    const latestPercent = project.context_usage?.latest_percent;
    context.textContent = latestPercent == null
        ? "Contexto sin datos"
        : `Contexto ${formatContextPercent(latestPercent)}`;
    context.title = project.context_usage?.maximum_percent == null
        ? "Todavía no hay consultas con telemetría de contexto."
        : `Máximo registrado ${formatContextPercent(
            project.context_usage.maximum_percent
        )} · media ${formatContextPercent(
            project.context_usage.average_percent
        )}`;
    openButton.append(name, count, context);
    openButton.addEventListener(
        "click",
        () => selectProject(project.id, project.name)
    );

    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "project-rename";
    renameButton.textContent = "✎";
    renameButton.title = "Renombrar proyecto";
    renameButton.setAttribute(
        "aria-label",
        `Renombrar proyecto ${project.name}`
    );
    renameButton.addEventListener("click", async () => {
        const requested = await askForText({
            title: "Renombrar proyecto",
            label: "Nombre del proyecto",
            value: project.name,
            confirmText: "Renombrar",
        });
        const newName = requested?.trim();
        if (!newName || newName === project.name) {
            return;
        }
        try {
            await apiRequest(
                `/api/projects/${encodeURIComponent(project.id)}`,
                {
                    method: "PATCH",
                    body: JSON.stringify({name: newName}),
                }
            );
            await loadProjects({loadThreads: false});
        } catch (error) {
            elements.queryStatus.textContent = error.message;
        }
    });

    item.append(openButton, renameButton);
    return item;
}

async function loadProjects({loadThreads = true} = {}) {
    try {
        const data = await apiRequest("/api/projects");
        const projects = data.projects || [];
        defaultProjectId = data.default_project_id ?? defaultProjectId;
        loadedProjects = projects;
        elements.projectList.replaceChildren();
        elements.projectListEmpty.classList.toggle(
            "hidden",
            projects.length > 0
        );
        projects.forEach((project) => {
            elements.projectList.append(createProjectListItem(project));
        });

        const savedProjectId = window.localStorage.getItem(
            ACTIVE_PROJECT_KEY
        );
        const activeStillExists = projects.some(
            (project) => project.id === activeProjectId
        );
        const savedStillExists = projects.some(
            (project) => project.id === savedProjectId
        );
        if (!activeStillExists) {
            activeProjectId = savedStillExists
                ? savedProjectId
                : defaultProjectId ?? projects[0]?.id ?? null;
        }
        if (activeProjectId) {
            window.localStorage.setItem(
                ACTIVE_PROJECT_KEY,
                activeProjectId
            );
        }
        markActiveProject();
        updateActiveProjectIndicators(projects);
        renderProjectAgentSettings(
            projects.find((project) => project.id === activeProjectId)
        );
        refreshContextIndicatorHeader();
        if (loadThreads && activeProjectId) {
            startNewConversation();
            await loadConversations({selectLatest: true});
        }
        return projects;
    } catch (error) {
        elements.projectListEmpty.textContent =
            `No se pudieron cargar los proyectos: ${error.message}`;
        elements.projectListEmpty.classList.remove("hidden");
        return [];
    }
}

function closeHistoryDrawer() {
    if (elements.historyDrawer) {
        elements.historyDrawer.open = false;
    }
}

async function openConversation(conversationId) {
    setMobileTab("chat");
    closeHistoryDrawer();
    try {
        const conversation = await apiRequest(
            `/api/conversations/${encodeURIComponent(conversationId)}`
        );
        renderConversation(conversation);
    } catch (error) {
        elements.queryStatus.textContent = error.message;
    }
}

function createConversationListItem(conversation) {
    const item = document.createElement("div");
    item.className = "conversation-item";
    item.dataset.conversationId = conversation.id;

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "conversation-open";
    openButton.title = conversation.title;
    const title = document.createElement("span");
    title.textContent = conversation.title;
    const count = document.createElement("small");
    count.textContent =
        `${conversation.turn_count} ` +
        (conversation.turn_count === 1 ? "consulta" : "consultas");
    openButton.append(title, count);
    openButton.addEventListener(
        "click",
        () => openConversation(conversation.id)
    );

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "conversation-delete";
    deleteButton.textContent = "×";
    deleteButton.title = "Borrar conversación";
    deleteButton.setAttribute(
        "aria-label",
        `Borrar conversación ${conversation.title}`
    );
    deleteButton.addEventListener("click", async () => {
        const confirmed = await askForConfirmation({
            title: "Borrar conversación",
            message:
                "Esta conversación se eliminará del historial del proyecto.",
            confirmText: "Borrar",
        });
        if (!confirmed) {
            return;
        }
        try {
            await apiRequest(
                `/api/conversations/${encodeURIComponent(conversation.id)}`,
                {method: "DELETE"}
            );
            if (activeConversationId === conversation.id) {
                startNewConversation();
            }
            await loadConversations({selectLatest: false});
            await loadProjects({loadThreads: false});
        } catch (error) {
            elements.queryStatus.textContent = error.message;
        }
    });

    item.append(openButton, deleteButton);
    return item;
}

async function loadConversations({selectLatest = false} = {}) {
    if (!activeProjectId) {
        elements.conversationList.replaceChildren();
        return [];
    }
    try {
        const data = await apiRequest(
            "/api/conversations?project_id=" +
            encodeURIComponent(activeProjectId)
        );
        const conversations = data.conversations || [];
        elements.conversationList.replaceChildren();
        elements.conversationListEmpty.classList.toggle(
            "hidden",
            conversations.length > 0
        );
        conversations.forEach((conversation) => {
            elements.conversationList.append(
                createConversationListItem(conversation)
            );
        });

        const active = conversations.find(
            (conversation) => conversation.id === activeConversationId
        );
        if (active) {
            activeConversationTitle = active.title;
            elements.conversationTitle.textContent = active.title;
        }
        markActiveConversation();
        updateHistorySummary(conversations.length);

        if (
            selectLatest
            && !activeConversationId
            && conversations.length > 0
        ) {
            await openConversation(conversations[0].id);
        }
        return conversations;
    } catch (error) {
        elements.conversationListEmpty.textContent =
            `No se pudo cargar el historial: ${error.message}`;
        elements.conversationListEmpty.classList.remove("hidden");
        return [];
    }
}

function updateHistorySummary(total) {
    if (!elements.historyDrawer) {
        return;
    }
    const summary = elements.historyDrawer.querySelector("summary");
    if (summary) {
        summary.textContent = total
            ? `Historial (${total})`
            : "Historial";
    }
}

function getSelectedTags() {
    return Array.from(
        elements.tagFilter.querySelectorAll('input[type="checkbox"]:checked')
    ).map((input) => input.value);
}

function updateTagFilterSummary() {
    const count = getSelectedTags().length;
    elements.tagFilterSummary.textContent = !count
        ? "Sin filtro"
        : count === 1
          ? "1 seleccionada"
          : `${count} seleccionadas`;
    elements.tagFilterClear.hidden = count === 0;
}

function filterTagOptions() {
    const query = elements.tagFilterSearch.value.trim().toLocaleLowerCase();
    const options = Array.from(
        elements.tagFilter.querySelectorAll(".tag-filter-option")
    );
    let visible = 0;
    for (const option of options) {
        const matches = option.dataset.tag.includes(query);
        option.hidden = !matches;
        if (matches) {
            visible += 1;
        }
    }
    elements.tagFilterEmpty.hidden = visible !== 0 || query.length === 0;
}

function clearTagFilter() {
    for (const input of elements.tagFilter.querySelectorAll(
        'input[type="checkbox"]:checked'
    )) {
        input.checked = false;
    }
    updateTagFilterSummary();
}

function createTagOption({tag, documents}, selected) {
    const option = document.createElement("label");
    option.className = "tag-filter-option";
    option.dataset.tag = tag.toLocaleLowerCase();

    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = tag;
    input.checked = selected;
    input.addEventListener("change", updateTagFilterSummary);

    const name = document.createElement("span");
    name.textContent = `#${tag}`;
    const count = document.createElement("span");
    count.className = "tag-filter-option-count";
    count.textContent = Number.isFinite(documents) ? String(documents) : "";
    count.hidden = !Number.isFinite(documents);
    option.append(input, name, count);
    return option;
}

function updateTagFilter(tags) {
    if (!Array.isArray(tags)) {
        return;
    }
    const selected = new Set(getSelectedTags());
    const normalizedTags = new Map();
    for (const entry of tags) {
        const tag = typeof entry === "string" ? entry : entry?.tag;
        if (typeof tag !== "string" || !tag) {
            continue;
        }
        const documents = Number(entry?.documents);
        normalizedTags.set(tag, {
            tag,
            documents: Number.isFinite(documents) ? documents : NaN,
        });
    }
    const tagOptions = [...normalizedTags.values()].sort((left, right) =>
        left.tag.localeCompare(right.tag, "es")
    );
    elements.tagFilter.replaceChildren(
        ...tagOptions.map((option) =>
            createTagOption(option, selected.has(option.tag))
        )
    );
    const hasTags = tagOptions.length > 0;
    elements.tagFilterControl.hidden = false;
    elements.tagFilterSearch.disabled = !hasTags;
    elements.tagFilterSearch.placeholder = hasTags
        ? "Buscar etiquetas"
        : "No hay etiquetas indexadas";
    elements.tagFilterSearch.value = "";
    elements.tagFilterEmpty.textContent = hasTags
        ? "No hay etiquetas que coincidan."
        : "Todavía no hay etiquetas indexadas.";
    elements.tagFilterEmpty.hidden = hasTags;
    updateTagFilterSummary();
    if (!hasTags) {
        elements.tagFilterSummary.textContent = "Sin etiquetas";
    }
}

elements.tagFilterSearch.addEventListener("input", filterTagOptions);
elements.tagFilterClear.addEventListener("click", clearTagFilter);

function updateStatusFilter(statuses) {
    // El vocabulario de estados lo define el vault del cliente, no una
    // lista escrita a mano en el HTML.
    if (!Array.isArray(statuses) || !statuses.length) {
        return;
    }
    const previous = elements.statusFilter.value;
    elements.statusFilter.replaceChildren();

    const all = document.createElement("option");
    all.value = "";
    all.textContent = "Todos";
    elements.statusFilter.append(all);

    for (const entry of statuses) {
        const option = document.createElement("option");
        option.value = entry.value;
        const label =
            entry.value.charAt(0).toUpperCase() + entry.value.slice(1);
        option.textContent = `${label} (${entry.documents})`;
        elements.statusFilter.append(option);
    }

    const stillAvailable = statuses.some(
        (entry) => entry.value === previous
    );
    elements.statusFilter.value = stillAvailable ? previous : "";
}

async function loadStatus() {
    try {
        const data = await apiRequest("/api/status");
        const stats = data.index?.stats ?? data.stats ?? {};
        updateCounts(stats);
        elements.vaultPath.textContent =
            data.vault_display_path ?? data.vault_path;

        if (
            !data.index
            || !data.ollama
            || !Array.isArray(data.chat_profiles)
            || !data.capabilities?.conversation_history
            || !data.capabilities?.projects
            || !data.capabilities?.conversation_memory
            || !data.capabilities?.hybrid_search
            || !data.capabilities?.project_agents
            || !data.capabilities?.project_context_usage
            || !data.capabilities?.project_memory
        ) {
            elements.selectVaultButton.disabled = true;
            elements.chatProfile.disabled = true;
            elements.systemMessage.textContent =
                "El servidor está ejecutando una versión anterior. " +
                "Cierra la ventana de iniciar_windows.bat, vuelve a " +
                "ejecutarlo y recarga esta página.";
            return null;
        }

        elements.selectVaultButton.disabled = false;
        elements.chatProfile.disabled = false;
        updateChatProfile(data);
        updateStatusFilter(data.document_statuses);
        updateTagFilter(
            data.retrieval?.vector_store?.tag_counts
            ?? data.retrieval?.vector_store?.tags
        );
        const graphConfigured =
            data.retrieval?.graph?.configured === true;
        elements.expandLinks.disabled = graphConfigured;
        if (graphConfigured) {
            elements.expandLinks.checked = false;
            elements.expandLinksLabel.textContent =
                "Seguir enlaces [[...]] (sustituido por el grafo)";
        } else {
            elements.expandLinksLabel.textContent =
                "Seguir enlaces [[...]]";
        }
        streamingSupported = data.capabilities?.streaming !== false;
        memorySummaryInterval =
            data.memory?.summary_interval ?? memorySummaryInterval;
        memoryRecentTurns =
            data.memory?.recent_turns ?? memoryRecentTurns;
        if (!activeConversationId) {
            updateMemoryStatus({enabled: elements.useMemory.checked});
        }
        elements.systemMessage.textContent = "";
        lastStatusData = data;
        refreshContextIndicatorHeader();
        return data;
    } catch (error) {
        elements.systemMessage.textContent = error.message;
        return null;
    }
}

elements.historyDrawer?.addEventListener("toggle", () => {
    // Un cajón abierto sobre el hilo estorba al escribir: se cierra al
    // enviar una pregunta.
    if (elements.historyDrawer.open) {
        elements.question.blur();
    }
});

elements.newConversationButton.addEventListener(
    "click",
    startNewConversation
);

elements.useMemory.addEventListener("change", async () => {
    const enabled = elements.useMemory.checked;
    if (!activeConversationId) {
        window.localStorage.setItem(
            MEMORY_PREFERENCE_KEY,
            String(enabled)
        );
        updateMemoryStatus({enabled});
        return;
    }

    elements.useMemory.disabled = true;
    elements.memoryStatus.textContent = "Guardando preferencia de memoria…";
    try {
        const result = await apiRequest(
            `/api/conversations/${encodeURIComponent(
                activeConversationId
            )}/memory`,
            {
                method: "PATCH",
                body: JSON.stringify({enabled}),
            }
        );
        window.localStorage.setItem(
            MEMORY_PREFERENCE_KEY,
            String(enabled)
        );
        updateMemoryStatus(result.memory);
    } catch (error) {
        elements.useMemory.checked = !enabled;
        elements.memoryStatus.textContent = error.message;
    } finally {
        elements.useMemory.disabled = false;
    }
});

elements.newProjectButton.addEventListener("click", async () => {
    const requested = await askForText({
        title: "Nuevo proyecto",
        label: "Nombre del proyecto",
        confirmText: "Crear",
    });
    const name = requested?.trim();
    if (!name) {
        return;
    }
    try {
        const project = await apiRequest("/api/projects", {
            method: "POST",
            body: JSON.stringify({name}),
        });
        activeProjectId = project.id;
        activeProjectName = project.name;
        await loadProjects({loadThreads: true});
    } catch (error) {
        elements.queryStatus.textContent = error.message;
    }
});

elements.deleteProjectButton.addEventListener("click", async () => {
    if (!activeProjectId || activeProjectId === defaultProjectId) {
        return;
    }
    const project = loadedProjects.find(
        (candidate) => candidate.id === activeProjectId
    );
    const projectName = project?.name || activeProjectName || "este proyecto";
    const conversations = Number(project?.conversation_count || 0);
    const conversationLabel =
        conversations === 1
            ? "1 conversacion"
            : `${conversations} conversaciones`;
    const confirmed = await askForConfirmation({
        title: "Eliminar proyecto",
        message:
            `Se eliminara "${projectName}". ` +
            `Sus ${conversationLabel} pasaran a General.`,
        confirmText: "Eliminar",
    });
    if (!confirmed) {
        return;
    }

    elements.deleteProjectButton.disabled = true;
    try {
        const result = await apiRequest(
            `/api/projects/${encodeURIComponent(activeProjectId)}`,
            {method: "DELETE"}
        );
        activeProjectId = result.fallback_project_id ?? defaultProjectId;
        activeProjectName = "";
        activeConversationId = null;
        activeConversationTitle = "";
        window.localStorage.setItem(ACTIVE_PROJECT_KEY, activeProjectId);
        elements.chatThread.replaceChildren();
        elements.useMemory.checked = preferredMemoryEnabled();
        updateMemoryStatus({enabled: elements.useMemory.checked});
        setChatLayout(false);
        await loadProjects({loadThreads: false});
        await loadConversations({selectLatest: false});
        elements.queryStatus.textContent =
            result.moved_conversations > 0
                ? `Proyecto eliminado. ${result.moved_conversations} ` +
                  "conversaciones movidas a General."
                : "Proyecto eliminado.";
    } catch (error) {
        elements.queryStatus.textContent = error.message;
    } finally {
        updateDeleteProjectButton();
    }
});

elements.projectVerificationEnabled.addEventListener("change", () => {
    elements.verifierContextTokens.disabled =
        !elements.projectVerificationEnabled.checked;
});

elements.verifierContextTokens.addEventListener("input", () => {
    updateContextValueReadout(
        elements.verifierContextTokens,
        elements.verifierContextValue
    );
});

elements.writerContextTokens.addEventListener("input", () => {
    updateContextValueReadout(
        elements.writerContextTokens,
        elements.writerContextValue
    );
});

elements.saveProjectAgentSettings.addEventListener("click", async () => {
    if (!activeProjectId) {
        return;
    }
    elements.saveProjectAgentSettings.disabled = true;
    elements.saveProjectAgentSettings.textContent = "Guardando…";
    try {
        await apiRequest(
            `/api/projects/${encodeURIComponent(
                activeProjectId
            )}/agent-settings`,
            {
                method: "PATCH",
                body: JSON.stringify({
                    verification_enabled:
                        elements.projectVerificationEnabled.checked,
                    project_memory_enabled:
                        elements.projectMemoryEnabled.checked,
                    verifier_context_tokens: Number(
                        elements.verifierContextTokens.value
                    ),
                    writer_context_tokens: Number(
                        elements.writerContextTokens.value
                    ),
                }),
            }
        );
        await loadProjects({loadThreads: false});
        elements.systemMessage.textContent =
            "Configuración de agentes guardada para el proyecto activo.";
    } catch (error) {
        elements.systemMessage.textContent = error.message;
    } finally {
        elements.saveProjectAgentSettings.disabled = false;
        elements.saveProjectAgentSettings.textContent = "Guardar cambios";
    }
});

elements.chatProfile.addEventListener("change", async () => {
    const profileId = elements.chatProfile.value;
    if (profileId === "custom") {
        return;
    }
    elements.chatProfile.disabled = true;
    elements.systemMessage.textContent = "Guardando perfil de respuesta…";
    try {
        const result = await apiRequest("/api/config/chat-profile", {
            method: "POST",
            body: JSON.stringify({profile: profileId}),
        });
        const status = await loadStatus();
        const installed = status?.ollama?.chat_model?.installed;
        const rerank = status?.retrieval?.rerank;
        const label = result.profile.label;
        const model = result.profile.model;
        // Que Ollama esté parado no es un problema del perfil elegido: ya
        // se ve en el resto de la interfaz en cuanto se intenta preguntar.
        // Repetirlo aquí sólo añade ruido. Que falte el modelo sí se avisa,
        // porque sin descargarlo la consulta fallará seguro.
        if (installed === false) {
            elements.systemMessage.textContent =
                `Perfil ${label} guardado, pero falta el modelo local. ` +
                `Ejecuta: ollama pull ${model}`;
        } else if (
            result.profile.rerank_enabled
            && (
                rerank?.model_available === false
                || rerank?.runtime_available === false
            )
        ) {
            elements.systemMessage.textContent =
                `Perfil ${label} guardado, pero falta el reranker ONNX local.`;
        } else {
            elements.systemMessage.textContent = "";
        }
    } catch (error) {
        await loadStatus();
        elements.systemMessage.textContent = error.message;
    } finally {
        elements.chatProfile.disabled = false;
    }
});

elements.selectVaultButton.addEventListener("click", async () => {
    elements.selectVaultButton.disabled = true;
    elements.selectVaultButton.textContent = "Seleccionando…";
    elements.systemMessage.textContent =
        "Selecciona la carpeta del vault en la ventana de Windows.";
    try {
        const result = await apiRequest("/api/vault/select", {
            method: "POST",
        });
        if (!result.selected) {
            elements.systemMessage.textContent =
                "No se cambió la carpeta del vault.";
            return;
        }
        elements.vaultPath.textContent =
            result.vault_display_path ?? result.vault_path;
        await loadStatus();
        elements.systemMessage.textContent =
            "Vault seleccionado. Pulsa «Indexar documentos» para " +
            "reconstruir o actualizar el índice.";
    } catch (error) {
        elements.systemMessage.textContent = error.message;
    } finally {
        elements.selectVaultButton.disabled = false;
        elements.selectVaultButton.textContent = "Elegir vault";
    }
});

function describeIndexResult(result) {
    const parts = [
        `${result.indexed} actualizados`,
        `${result.unchanged} sin cambios`,
        `${result.deleted} eliminados`,
        `${result.chunks_created} fragmentos nuevos`,
    ];
    if (result.errors.length) {
        parts.push(`${result.errors.length} errores`);
        parts.push(result.errors[result.errors.length - 1]);
        console.error(result.errors);
    }
    if (result.rebuilt) {
        parts.push(
            `índice reconstruido: ${result.rebuild_reasons.join(", ")}`
        );
    }
    return parts.join(" · ");
}

function describeIndexProgress(progress) {
    if (progress.phase === "unchanged" || progress.phase === "embedding") {
        const position = progress.total
            ? `${progress.processed} de ${progress.total}`
            : `${progress.processed}`;
        const file = progress.current_file
            ? ` · ${progress.current_file}`
            : "";
        return `Indexando ${position} archivos${file}`;
    }
    if (progress.phase === "scanning") {
        return "Explorando el vault…";
    }
    return "Preparando la indexación…";
}

async function pollIndexingProgress() {
    while (true) {
        await new Promise((resolve) => setTimeout(resolve, 400));
        let progress;
        try {
            progress = await apiRequest("/api/index/progress");
        } catch (error) {
            throw new Error(
                "Se perdió el contacto con el servidor durante la " +
                "indexación."
            );
        }

        if (progress.running) {
            elements.systemMessage.textContent =
                describeIndexProgress(progress);
            continue;
        }
        if (progress.error) {
            throw new Error(progress.error);
        }
        if (progress.result) {
            return progress.result;
        }
        throw new Error("La indexación terminó sin resultado.");
    }
}

elements.indexButton.addEventListener("click", async () => {
    elements.indexButton.disabled = true;
    elements.indexButton.textContent = "Indexando…";
    elements.systemMessage.textContent = "Preparando la indexación…";
    try {
        await apiRequest("/api/index/start", {
            method: "POST",
            body: JSON.stringify({}),
        });
        const result = await pollIndexingProgress();
        updateCounts(result.stats);
        elements.systemMessage.textContent = describeIndexResult(result);
        await loadStatus();
    } catch (error) {
        elements.systemMessage.textContent = error.message;
    } finally {
        elements.indexButton.disabled = false;
        elements.indexButton.textContent = "Indexar documentos";
    }
});

elements.stopQueryButton.addEventListener("click", () => {
    if (!activeQueryController) {
        return;
    }
    elements.stopQueryButton.disabled = true;
    elements.queryStatus.textContent = "Deteniendo consulta…";
    activeQueryController.abort();
});

elements.question.addEventListener("keydown", (event) => {
    // Enter envía la pregunta; Mayús+Enter sigue insertando un salto de
    // línea. isComposing evita enviar a medias mientras un IME (chino,
    // japonés, coreano...) todavía está componiendo el carácter.
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) {
        return;
    }
    event.preventDefault();
    if (elements.askButton.disabled) {
        return;
    }
    if (typeof elements.form.requestSubmit === "function") {
        elements.form.requestSubmit(elements.askButton);
    } else {
        elements.askButton.click();
    }
});

elements.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = elements.question.value.trim();
    if (!question) {
        return;
    }

    closeHistoryDrawer();
    setMobileTab("chat");
    activeQueryController = new AbortController();
    setQueryRunning(true);
    elements.queryStatus.textContent = "Consultando la documentación…";
    elements.question.value = "";

    const body = {
        question,
        top_k: Number(elements.topK.value),
        status: elements.statusFilter.value || null,
        vigencia: elements.vigenciaFilter.value || null,
        tags: getSelectedTags(),
        expand_links: elements.expandLinks.checked,
        conversation_id: activeConversationId,
        project_id: activeProjectId,
        use_memory: elements.useMemory.checked,
    };

    let pendingTurn = null;

    try {
        let result = null;

        if (streamingSupported) {
            result = await streamQuery(
                question,
                body,
                activeQueryController.signal
            );
        }

        if (result === null) {
            // Vía clásica: sin streaming o no disponible en el servidor.
            pendingTurn = createTurnElement(
                {question, sources: []},
                {pending: true}
            );
            elements.chatThread.append(pendingTurn);
            setChatLayout(
                true,
                activeConversationTitle || "Nueva conversación"
            );
            scrollToLatestTurn();

            result = await apiRequest("/api/query", {
                method: "POST",
                body: JSON.stringify(body),
                signal: activeQueryController.signal,
            });
            const completedTurn = createTurnElement({
                id: result.turn_id,
                question,
                answer: result.answer,
                sources: result.sources,
                chat_model: result.chat_model,
                agents: result.agents,
            });
            pendingTurn.replaceWith(completedTurn);
            pendingTurn = null;
        }

        elements.stopQueryButton.hidden = true;
        activeConversationId = result.conversation_id;
        await loadConversations();
        await loadProjects({loadThreads: false});
        setChatLayout(true, activeConversationTitle);
        updateMemoryStatus(result.memory);
        latestTurnMaxUsagePercent = maxUsagePercentFromAgents(result.agents);
        refreshContextIndicatorHeader();

        const warnings = [
            result.agents?.warning,
            result.memory?.warning,
            result.project_memory?.warning,
        ].filter(Boolean);
        elements.queryStatus.textContent = warnings.length
            ? `Consulta terminada con avisos: ${warnings.join(" ")}`
            : "Consulta terminada.";
        scrollToLatestTurn();
    } catch (error) {
        const stopped = isAbortError(error);
        const failedTurn = createTurnElement(
            {question, sources: []},
            {
                error: stopped
                    ? "Consulta detenida por el usuario."
                    : error.message,
            }
        );
        if (pendingTurn) {
            pendingTurn.replaceWith(failedTurn);
        } else {
            elements.chatThread.append(failedTurn);
        }
        setChatLayout(true, activeConversationTitle || "Nueva conversación");
        elements.queryStatus.textContent = stopped
            ? "Consulta detenida."
            : "No se pudo completar la consulta.";
        scrollToLatestTurn();
    } finally {
        activeQueryController = null;
        setQueryRunning(false);
        elements.question.focus();
    }
});

Promise.all([
    loadStatus(),
    loadProjects({loadThreads: true}),
]);
