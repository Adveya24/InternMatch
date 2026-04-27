const STORAGE_KEYS = { lastUpload: "internmatch:lastUpload" };
const APP_VERSION = "v2.2";

// Clear stale cached results if app version changed
(function bustCache() {
  const storedVersion = localStorage.getItem("internmatch:version");
  if (storedVersion !== APP_VERSION) {
    localStorage.removeItem(STORAGE_KEYS.lastUpload);
    localStorage.setItem("internmatch:version", APP_VERSION);
  }
})();


function $(id) {
  return document.getElementById(id);
}

async function updateModeBanner(mode) {
  const banner = $("mode-banner");
  if (!banner) return;

  let resolvedMode = mode;
  if (!resolvedMode) {
    try {
      const response = await fetch("/health");
      const data = await response.json();
      resolvedMode = data.mode || (data.ollama_enabled ? "ai" : "fallback");
    } catch (_err) {
      resolvedMode = "fallback";
    }
  }

  banner.textContent =
    resolvedMode === "ai"
      ? "Cloud logic active."
      : "AI engine unavailable, showing basic matches.";
  banner.hidden = resolvedMode === "ai";
}

async function loadDepartments() {
  const grid = $("departments-grid");
  const select = $("departments");
  if (!grid && !select) return;

  const response = await fetch("/skills");
  const taxonomy = await response.json();
  Object.keys(taxonomy).forEach((department) => {
    if (grid) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "department-option";
      button.dataset.department = department;
      button.innerHTML = "<span class='department-name'>" + department + "</span>";
      button.addEventListener("click", () => toggleDepartment(button));
      grid.appendChild(button);
    } else if (select) {
      const option = document.createElement("option");
      option.value = department;
      option.textContent = department;
      select.appendChild(option);
    }
  });

  updateDepartmentNote();
}

function getSelectedDepartments() {
  const grid = $("departments-grid");
  const select = $("departments");
  if (grid) {
    return Array.from(grid.querySelectorAll(".department-option.selected")).map((button) => button.dataset.department);
  }
  if (!select) return [];
  return Array.from(select.selectedOptions).map((option) => option.value).slice(0, 1);
}

function updateDepartmentNote() {
  const note = $("department-note");
  const grid = $("departments-grid");
  if (!note || !grid) return;

  const selected = getSelectedDepartments();
  note.textContent =
    selected.length === 0
      ? "No departments selected (Global Match enabled)."
      : "1 selected.";
}

function toggleDepartment(button) {
  const isSelected = button.classList.contains("selected");
  
  // Clear all other selections
  const grid = $("departments-grid");
  if (grid) {
    Array.from(grid.querySelectorAll(".department-option.selected")).forEach((b) => b.classList.remove("selected"));
  }

  if (!isSelected) {
    button.classList.add("selected");
  }
  
  updateDepartmentNote();
}

function renderSkills(skills) {
  const card = $("skills-preview-card");
  const list = $("skills-list");
  if (!card || !list) return;

  list.innerHTML = "";
  (skills || []).forEach((skill) => {
    const item = document.createElement("li");
    item.textContent = skill.name + " (" + skill.confidence.toFixed(2) + ")";
    list.appendChild(item);
  });
  card.hidden = !skills || skills.length === 0;
}

function renderManualSkills(skills) {
  const card = $("manual-skills-card");
  const list = $("manual-skills-list");
  if (!card || !list) return;

  list.innerHTML = "";
  skills.forEach((skill) => {
    const label = document.createElement("label");
    label.className = "skill-chip";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = skill;

    const text = document.createElement("span");
    text.textContent = skill;

    label.appendChild(checkbox);
    label.appendChild(text);
    list.appendChild(label);
  });

  card.hidden = skills.length === 0;
}

function selectedManualSkills() {
  const list = $("manual-skills-list");
  if (!list) return [];
  return Array.from(list.querySelectorAll("input[type='checkbox']:checked")).map((input) => input.value);
}

function renderPdfPreview(file) {
  const card = $("pdf-preview-card");
  const canvas = $("pdf-canvas");
  if (!card || !canvas || !file || !file.type.includes("pdf") || !window.pdfjsLib) return;

  const reader = new FileReader();
  reader.onload = function (event) {
    const typedArray = new Uint8Array(event.target.result);
    window.pdfjsLib.getDocument({ data: typedArray }).promise.then((pdf) => {
      pdf.getPage(1).then((page) => {
        const viewport = page.getViewport({ scale: 1.15 });
        const context = canvas.getContext("2d");
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        page.render({ canvasContext: context, viewport: viewport });
        card.hidden = false;
      });
    });
  };
  reader.readAsArrayBuffer(file);
}

function updateFileLabel(file) {
  const label = $("file-label");
  if (!label) return;
  label.textContent = file ? file.name : "Choose a CV file";
}

async function requestMatches(payload) {
  const response = await fetch("/matches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Matching failed.");
  }

  updateModeBanner(data.mode);
  return data;
}

async function fetchMatchesAndNavigate(manualSkills) {
  const raw = localStorage.getItem(STORAGE_KEYS.lastUpload);
  if (!raw) return;

  const payload = JSON.parse(raw);
  const data = await requestMatches({
    cv_path: payload.cvPath,
    departments: payload.departments,
    manual_skills: manualSkills || [],
  });

  if (data.suggested_skills && data.suggested_skills.length > 0) {
    renderManualSkills(data.suggested_skills.slice(0, 40));
    $("status").textContent = data.message;
    return;
  }

  payload.matches = data.matches || [];
  payload.best_match = data.best_match || null;
  payload.mode = data.mode || "fallback";
  payload.source = data.source || "fallback";
  payload.aiSource = data.ai_source || "Cloud Engine (Prototype)";
  payload.skills = data.skills_detected || payload.skills || [];
  payload.experienceSummary = data.experience_summary || "";
  payload.educationSummary = data.education_summary || "";
  payload.projectsSummary = data.projects_summary || "";
  payload.summary = data.summary || "";
  payload.cv_review = data.cv_review || "";
  payload.message = data.message || "";
  payload.parsedExperienceSection = data.parsed_experience_section || "";
  payload.parsedEducationSection = data.parsed_education_section || "";
  payload.parsedProjectsSection = data.parsed_projects_section || "";
  payload.cv_rating = data.cv_rating || 5;
  localStorage.setItem(STORAGE_KEYS.lastUpload, JSON.stringify(payload));
  window.location.href = "/results";
}

let _loadingTimer = null;
let _loadingStart = 0;

function showLoading(show) {
  const container = $("loading-container");
  const btn = $("submit-btn");
  if (container) container.hidden = !show;
  if (btn) btn.disabled = show;
  if (show) {
    _loadingStart = Date.now();
    const bar = $("loading-progress");
    if (bar) { bar.style.width = "0%"; void bar.offsetWidth; bar.style.width = "90%"; }
    _loadingTimer = setInterval(() => {
      const el = $("loading-phase");
      if (el) {
        const secs = Math.floor((Date.now() - _loadingStart) / 1000);
        const phase = el.dataset.phase || "Working";
        el.textContent = phase + " (" + secs + "s elapsed)";
      }
    }, 1000);
  } else {
    if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
  }
}

function setLoadingPhase(phase) {
  const el = $("loading-phase");
  if (el) { el.dataset.phase = phase; el.textContent = phase; }
}

async function handleCvFormSubmit(event) {
  event.preventDefault();
  const fileInput = $("cv-file");
  const statusEl = $("status");
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    statusEl.textContent = "Choose a CV file first.";
    return;
  }

  const departments = getSelectedDepartments();
  // We no longer require department length > 0. A length of 0 engages global matching mode.

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  statusEl.textContent = "";
  showLoading(true);
  setLoadingPhase("Uploading CV...");

  try {
    setLoadingPhase("Parsing CV text...");
    const response = await fetch("/upload-cv", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      showLoading(false);
      statusEl.textContent = data.error || "Upload failed.";
      return;
    }

    setLoadingPhase("CV parsed ✓  Searching live internship APIs...");
    renderSkills(data.skills || []);
    localStorage.setItem(
      STORAGE_KEYS.lastUpload,
      JSON.stringify({
        cvPath: data.path,
        cvId: data.cv_id,
        departments: departments,
        skills: data.skills || [],
        source: data.source || "local-parse",
        aiSource: data.ai_source || "Cloud Engine (Prototype)",
        experienceSummary: data.experience_summary || "",
        educationSummary: data.education_summary || "",
        projectsSummary: data.projects_summary || "",
        summary: data.summary || "",
        cv_review: data.cv_review || "",
        message: data.message || "",
        parsedExperienceSection: data.parsed_experience_section || "",
        parsedEducationSection: data.parsed_education_section || "",
        parsedProjectsSection: data.parsed_projects_section || "",
      })
    );

    setLoadingPhase("AI is ranking matches (this takes a moment)...");
    await fetchMatchesAndNavigate([]);
  } catch (err) {
    console.error(err);
    showLoading(false);
    statusEl.textContent = "Something went wrong while processing the CV.";
  }
}

async function handleManualSkillsSubmit() {
  const statusEl = $("status");
  const skills = selectedManualSkills();
  if (skills.length === 0) {
    statusEl.textContent = "Select at least one skill before retrying.";
    return;
  }

  statusEl.textContent = "Ranking internships with manual skills...";
  try {
    await fetchMatchesAndNavigate(skills);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Matching failed with manual skills.";
  }
}

function renderLearningResources(resources) {
  return ""; // Deprecated: Replaced by text-based AI learning suggestions section
}

function renderProfilePanel(payload) {
  const panel = $("profile-panel");
  if (!panel) return;

  const truncate = (text, len) => {
    if (!text) return "Not provided.";
    if (text.length <= len) return text;
    const safeText = text.replace(/>/g, "&gt;").replace(/</g, "&lt;");
    const short = safeText.slice(0, len) + "...";
    return `<span class="trunc-short">${short} <button type="button" class="btn-link" onclick="this.parentElement.hidden=true; this.parentElement.nextElementSibling.hidden=false;">Show more</button></span><span class="trunc-full" hidden>${safeText} <button type="button" class="btn-link" onclick="this.parentElement.hidden=true; this.parentElement.previousElementSibling.hidden=false;">Show less</button></span>`;
  };

  const summaryHtml = `<div class='profile-section'><h4>Summary</h4><p>${truncate(payload.summary, 220)}</p></div>`;

  panel.innerHTML =
    summaryHtml +
    `<div class='profile-section'><h4>Experience</h4><p>${truncate(payload.experienceSummary, 150)}</p></div>` +
    `<div class='profile-section'><h4>Education</h4><p>${truncate(payload.educationSummary, 150)}</p></div>` +
    `<div class='profile-section'><h4>Projects</h4><p>${truncate(payload.projectsSummary, 150)}</p></div>`;
}

function renderResultsPage() {
  const summaryEl = $("summary");
  const aiSourceEl = $("ai-source-label");
  const matchesEl = $("matches");
  const bestMatchPanel = $("best-match-panel");
  const profilePanel = $("profile-panel");
  if (!summaryEl || !matchesEl) return;

  const raw = localStorage.getItem(STORAGE_KEYS.lastUpload);
  if (!raw) {
    summaryEl.textContent = "No recent CV upload found. Go back and upload a CV.";
    return;
  }

  const payload = JSON.parse(raw);
  window.internshipData = {}; // clear global dictionary mapping for chats
  updateModeBanner(payload.mode);

  summaryEl.textContent =
    "CV: " +
    (payload.cvId || "unknown") +
    " | Departments: " +
    (payload.departments || []).join(", ") +
    " | Internship source: " +
    (payload.source || "unavailable") +
    " | AI Match Percentage generated via LLM reasoning.";
  if (aiSourceEl) {
    aiSourceEl.textContent = "AI Source: " + (payload.aiSource || "Cloud Engine (Prototype)");
  }

  if (profilePanel) {
    renderProfilePanel(payload);
  }

  // CV Rating Widget
  const cvScore = payload.cv_rating || 5;
  const cvScoreEl = document.getElementById("cv-score");
  const cvRingEl = document.getElementById("cv-ring");
  const cvLabelEl = document.getElementById("cv-rating-label");
  if (cvScoreEl && cvRingEl && cvLabelEl) {
    cvScoreEl.textContent = cvScore + "/10";
    const pct = cvScore * 10;
    let color = "#e74c3c";
    let label = "Needs significant improvement";
    if (cvScore >= 8) { color = "#27ae60"; label = "Excellent CV for your level"; }
    else if (cvScore >= 6) { color = "#f39c12"; label = "Good foundation, room to grow"; }
    else if (cvScore >= 4) { color = "#e67e22"; label = "Average — add more projects & skills"; }
    cvRingEl.style.background = `conic-gradient(${color} ${pct}%, #e0e0e0 ${pct}%)`;
    cvLabelEl.textContent = label;
  }

  if (bestMatchPanel && payload.best_match) {
    const internship = payload.best_match.internship || {};
    bestMatchPanel.innerHTML =
      "<p class='spotlight-title'>" +
      (internship.title || "Unknown role") +
      "</p>" +
      "<p>" +
      (internship.company || "Unknown company") +
      " | " +
      (internship.department || "") +
      "</p>" +
      "<p>" +
      (payload.best_match.matched_skills.join(", ") || "No matched skills listed") +
      "</p>" +
      "<p>" +
      (payload.best_match.learning_suggestion || "Keep building relevant project depth.") +
      "</p>" +
      "<span class='spotlight-score'>" +
      payload.best_match.final_score +
      "% match</span>";
  }

  matchesEl.innerHTML = "";
  const matches = payload.matches || [];
  if (matches.length === 0) {
    matchesEl.innerHTML =
      "<div class='empty-state'>" +
      "<p><strong>" +
      (payload.message || "No matches available.") +
      "</strong></p>" +
      "<p><strong>Parsed experience section:</strong> " +
      (payload.parsedExperienceSection || "Not available.") +
      "</p>" +
      "<p><strong>Parsed education section:</strong> " +
      (payload.parsedEducationSection || "Not available.") +
      "</p>" +
      "<p><strong>Parsed projects section:</strong> " +
      (payload.parsedProjectsSection || "Not available.") +
      "</p>" +
      "</div>";
    return;
  }

  matches.forEach((match, index) => {
    const internship = match.internship || {};
    window.internshipData[index] = internship;
    const card = document.createElement("article");
    card.className = "match";

    const applyUrl = internship.apply_url || "#";
    // Map source to human-readable label; hide 'CSV Fallback' entirely
    const rawSource = (internship.source || "").trim();
    const sourceLabel = rawSource && rawSource.toLowerCase() !== "csv fallback" && rawSource !== "" 
      ? rawSource.charAt(0).toUpperCase() + rawSource.slice(1) 
      : "Live API";

    // Build learning resource links for missing skills
    const resources = match.learning_resources || {};
    let resourcesHtml = "";
    const missingSkills = match.missing_skills || [];
    if (missingSkills.length > 0) {
      resourcesHtml = '<div class="missing-skill-courses"><strong>Courses for missing skills:</strong><ul>';
      missingSkills.forEach(skill => {
        const urls = resources[skill] || [];
        if (urls.length > 0) {
          const links = urls.map(u => `<a href="${u}" target="_blank" rel="noopener">${new URL(u).hostname.replace('www.','')}</a>`).join(", ");
          resourcesHtml += `<li><strong>${skill}:</strong> ${links}</li>`;
        } else {
          resourcesHtml += `<li><strong>${skill}:</strong> <a href="${urls[0]}" target="_blank" rel="noopener">Find courses</a></li>`;
        }
      });
      resourcesHtml += '</ul></div>';
    }

    const suggestions = match.learning_suggestions && match.learning_suggestions.length
      ? match.learning_suggestions.join(", ")
      : "Build deeper project evidence for the missing skills.";

    card.innerHTML = `
      <div class="match-head">
        <div class="match-title">
          <h3>${index === 0 ? "Top match: " : ""}${internship.title || "Unknown role"}</h3>
          <p class="meta">
            <span>${internship.company || "Unknown company"}</span>
            <span>${internship.location || "Unknown location"}</span>
          </p>
        </div>
        <div class="match-actions">
          <span class="score">${match.final_score}%</span>
          <a href="${applyUrl}" target="_blank" rel="noopener noreferrer" class="apply-btn">Apply</a>
        </div>
      </div>
      <div class="match-body">
        <div class="match-tags">
          <span class="tag-item"><strong>Source:</strong> ${sourceLabel}</span>
          <span class="tag-item competitiveness-${(match.competitiveness || 'Medium').toLowerCase()}">
            <strong>Competitiveness:</strong> ${match.competitiveness || 'Medium'}
          </span>
        </div>
        <p><strong>Matched skills:</strong> ${match.matched_skills.join(", ") || "None"}</p>
        <p><strong>Missing skills:</strong> ${missingSkills.join(", ") || "None"}</p>
        <p><strong>Experience Alignment:</strong> ${match.experience_alignment || ""}</p>
        <div class="ai-insights">
          <p class="explanation"><strong>Explanation:</strong> ${match.explanation}</p>
          <p><strong>Suggestions:</strong> ${suggestions}</p>
          ${resourcesHtml}
        </div>
        <div class="chat-section" style="margin-top: 15px; border-top: 1px solid #ddd; padding-top: 10px;">
          <input type="text" class="chat-input" id="chat-input-${index}" placeholder="Ask a question about this job..." style="width: 70%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;" />
          <button class="apply-btn" type="button" onclick="handleJobChat(${index})" style="padding: 8px 16px; margin-left: 5px; cursor: pointer;">Ask</button>
          <div class="chat-response" id="chat-response-${index}" style="margin-top: 10px; font-style: italic; color: #555;"></div>
        </div>
      </div>
    `;
    matchesEl.appendChild(card);
  });
}

window.handleJobChat = async function(index) {
    const input = document.getElementById(`chat-input-${index}`);
    const responseEl = document.getElementById(`chat-response-${index}`);
    const question = input.value.trim();
    if (!question) return;
    const jobData = window.internshipData[index];
    const raw = localStorage.getItem(STORAGE_KEYS.lastUpload);
    if (!raw) return;
    const payload = JSON.parse(raw);
    responseEl.textContent = "Thinking...";
    try {
        const response = await fetch("/ask-job-question", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cv_path: payload.cvPath, job: jobData, question: question })
        });
        const data = await response.json();
        responseEl.innerHTML = response.ok ? `<strong>A:</strong> ${data.answer}` : "Error: " + data.error;
    } catch (err) { responseEl.textContent = "Failed to get an answer."; }
    input.value = "";
};

function init() {
  const form = $("cv-form");
  updateModeBanner();

  if (form) {
    form.addEventListener("submit", handleCvFormSubmit);
    loadDepartments().catch((err) => {
      console.error(err);
      $("status").textContent = "Failed to load department list.";
    });

    const fileInput = $("cv-file");
    if (fileInput) {
      fileInput.addEventListener("change", () => {
        const file = fileInput.files && fileInput.files[0];
        updateFileLabel(file);
        renderPdfPreview(file);
      });
    }

    const manualButton = $("manual-skills-submit");
    if (manualButton) {
      manualButton.addEventListener("click", handleManualSkillsSubmit);
    }
  } else {
    renderResultsPage();
  }
}

document.addEventListener("DOMContentLoaded", init);
