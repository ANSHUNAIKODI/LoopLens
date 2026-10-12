/**
 * LoopLens - Client Application
 * Communicates with LoopLens FastAPI backend:
 * 1. YouTube Community Analysis (Metadata & Public Comments via YouTube Data API v3)
 * 2. Video Frame Repetition Analysis (Local Computer-Vision Perceptual Hashing Prototype)
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- Global / Header Elements ---
  const statusBadge = document.getElementById("api-status-badge");
  const statusText = document.getElementById("api-status-text");
  const apiKeyBanner = document.getElementById("api-key-banner");

  // --- Mode Switcher Elements ---
  const tabYoutube = document.getElementById("tab-youtube");
  const tabVideo = document.getElementById("tab-video");
  const panelYoutube = document.getElementById("panel-youtube");
  const panelVideo = document.getElementById("panel-video");

  // --- YouTube Mode Elements ---
  const ytForm = document.getElementById("analyze-form");
  const urlInput = document.getElementById("video-url-input");
  const clearBtn = document.getElementById("clear-input-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnIcon = document.getElementById("btn-icon");
  const btnText = document.getElementById("btn-text");
  const inputError = document.getElementById("input-error");

  // --- Video Upload Mode Elements ---
  const uploadForm = document.getElementById("upload-video-form");
  const dropzone = document.getElementById("video-dropzone");
  const videoFileInput = document.getElementById("video-file-input");
  const selectedFileCard = document.getElementById("selected-file-card");
  const selectedFileName = document.getElementById("selected-file-name");
  const selectedFileSize = document.getElementById("selected-file-size");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const permissionCheckbox = document.getElementById("permission-checkbox");
  const sampleIntervalSelect = document.getElementById("sample-interval-select");
  const minSequenceSelect = document.getElementById("min-sequence-select");
  const uploadAnalyzeBtn = document.getElementById("upload-analyze-btn");
  const uploadBtnSpinner = document.getElementById("upload-btn-spinner");
  const uploadBtnIcon = document.getElementById("upload-btn-icon");
  const uploadBtnText = document.getElementById("upload-btn-text");
  const runDemoBtn = document.getElementById("run-demo-btn");
  const demoTypeSelect = document.getElementById("demo-type-select");
  const videoInputError = document.getElementById("video-input-error");

  // --- State View Elements ---
  const loadingView = document.getElementById("loading-view");
  const loadingStepText = document.getElementById("loading-step-text");
  const errorView = document.getElementById("error-view");
  const errorTitle = document.getElementById("error-title");
  const errorDesc = document.getElementById("error-description");
  const errorDismissBtn = document.getElementById("error-dismiss-btn");

  // --- YouTube Results View Elements ---
  const resultsView = document.getElementById("results-view");
  const videoThumb = document.getElementById("video-thumbnail");
  const videoDurationBadge = document.getElementById("video-duration-badge");
  const videoPublishedDate = document.getElementById("video-published-date");
  const videoTitle = document.getElementById("video-title");
  const videoChannel = document.getElementById("video-channel");
  const videoViews = document.getElementById("video-views-count");
  const videoDurationText = document.getElementById("video-duration-text");
  const videoComments = document.getElementById("video-comments-count");

  const verdictCard = document.getElementById("verdict-card");
  const verdictIconBox = document.getElementById("verdict-icon-box");
  const verdictTag = document.getElementById("verdict-tag");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictSummary = document.getElementById("verdict-summary");
  const verdictNotice = document.getElementById("verdict-notice-text");

  const metricAnalyzed = document.getElementById("metric-analyzed");
  const metricFlagged = document.getElementById("metric-flagged");
  const metricRepeat = document.getElementById("metric-repeat-reports");
  const metricRecap = document.getElementById("metric-recap-reports");
  const metricIncomplete = document.getElementById("metric-incomplete-reports");

  const pipeMetaStatus = document.getElementById("pipe-meta-status");
  const pipeMetaDesc = document.getElementById("pipe-meta-desc");
  const pipeCommunityStatus = document.getElementById("pipe-community-status");
  const pipeCommunityDesc = document.getElementById("pipe-community-desc");
  const pipeFootageStatus = document.getElementById("pipe-footage-status");
  const pipeFootageDesc = document.getElementById("pipe-footage-desc");

  const timestampsCard = document.getElementById("timestamps-card");
  const timestampsList = document.getElementById("timestamps-list");
  const evidenceCommentsList = document.getElementById("evidence-comments-list");
  const resetBtn = document.getElementById("reset-analysis-btn");

  // YouTube Visual Footage Evidence Elements
  const ytVisualSection = document.getElementById("yt-visual-analysis-section");
  const ytMetricVideoTotalMatches = document.getElementById("yt-metric-video-total-matches");
  const ytMetricVideoDynamicMatches = document.getElementById("yt-metric-video-dynamic-matches");
  const ytMetricVideoStaticMatches = document.getElementById("yt-metric-video-static-matches");
  const ytMetricVideoRepeatSec = document.getElementById("yt-metric-video-repeat-sec");
  const ytVideoMatchesList = document.getElementById("yt-video-matches-list");

  // --- Video Frame Analysis Results View Elements ---
  const videoResultsView = document.getElementById("video-results-view");
  const fileDisplayName = document.getElementById("file-display-name");
  const fileDurationBadge = document.getElementById("file-duration-badge");
  const fileSpecsSummary = document.getElementById("file-specs-summary");
  const fileDurationText = document.getElementById("file-duration-text");
  const fileSampledCount = document.getElementById("file-sampled-count");
  const fileIntervalText = document.getElementById("file-interval-text");
  const fileTotalFrames = document.getElementById("file-total-frames");

  const videoVerdictCard = document.getElementById("video-verdict-card");
  const videoVerdictIconBox = document.getElementById("video-verdict-icon-box");
  const videoVerdictTag = document.getElementById("video-verdict-tag");
  const videoVerdictTitle = document.getElementById("video-verdict-title");
  const videoVerdictSummary = document.getElementById("video-verdict-summary");
  const videoVerdictNotice = document.getElementById("video-verdict-notice-text");

  const metricVideoTotalMatches = document.getElementById("metric-video-total-matches");
  const metricVideoDynamicMatches = document.getElementById("metric-video-dynamic-matches");
  const metricVideoStaticMatches = document.getElementById("metric-video-static-matches");
  const metricVideoRepeatSec = document.getElementById("metric-video-repeat-sec");
  const videoMatchesList = document.getElementById("video-matches-list");
  const resetVideoAnalysisBtn = document.getElementById("reset-video-analysis-btn");

  // Configuration & State
  const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB
  const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
  let selectedFile = null;
  let currentActiveMode = "youtube"; // "youtube" | "video"
  let loadingInterval = null;

  // Initialize System Status Check
  checkServerStatus();

  // --- Mode Navigation ---
  tabYoutube.addEventListener("click", () => switchMode("youtube"));
  tabVideo.addEventListener("click", () => switchMode("video"));

  function switchMode(mode) {
    if (currentActiveMode === mode) return;
    currentActiveMode = mode;
    hideAllStates();

    if (mode === "youtube") {
      tabYoutube.classList.add("active");
      tabYoutube.setAttribute("aria-selected", "true");
      tabVideo.classList.remove("active");
      tabVideo.setAttribute("aria-selected", "false");

      panelYoutube.style.display = "block";
      panelVideo.style.display = "none";
      urlInput.focus();
    } else {
      tabVideo.classList.add("active");
      tabVideo.setAttribute("aria-selected", "true");
      tabYoutube.classList.remove("active");
      tabYoutube.setAttribute("aria-selected", "false");

      panelVideo.style.display = "block";
      panelYoutube.style.display = "none";
    }
  }

  // --- YouTube Input & Clear ---
  urlInput.addEventListener("input", () => {
    hideInputError();
    clearBtn.style.display = urlInput.value.trim() ? "block" : "none";
  });

  clearBtn.addEventListener("click", () => {
    urlInput.value = "";
    clearBtn.style.display = "none";
    hideInputError();
    urlInput.focus();
  });

  // --- YouTube Form Submit ---
  ytForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const rawUrl = urlInput.value.trim();

    if (!rawUrl) {
      showInputError("Please enter a YouTube video URL or ID.");
      urlInput.focus();
      return;
    }

    if (!isValidYouTubeUrlOrId(rawUrl)) {
      showInputError("Please provide a valid YouTube URL (e.g. https://www.youtube.com/watch?v=... or youtu.be/...)");
      urlInput.focus();
      return;
    }

    hideInputError();
    await performAnalysis(rawUrl);
  });

  // --- Video File Drag & Drop & Selection ---
  dropzone.addEventListener("click", () => {
    videoFileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      videoFileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  removeFileBtn.addEventListener("click", () => {
    clearSelectedFile();
  });

  permissionCheckbox.addEventListener("change", () => {
    updateUploadButtonState();
    hideVideoInputError();
  });

  function handleFileSelection(file) {
    hideVideoInputError();
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      showVideoInputError(
        `Unsupported video format '${ext}'. Please choose an MP4, WebM, MOV, AVI, or MKV file.`
      );
      clearSelectedFile();
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      showVideoInputError(
        `Selected file exceeds the 100 MB limit (${sizeMB} MB). Please choose a shorter clip or trim the file.`
      );
      clearSelectedFile();
      return;
    }

    selectedFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);
    selectedFileCard.style.display = "flex";
    dropzone.style.display = "none";
    updateUploadButtonState();
  }

  function clearSelectedFile() {
    selectedFile = null;
    videoFileInput.value = "";
    selectedFileCard.style.display = "none";
    dropzone.style.display = "block";
    updateUploadButtonState();
    hideVideoInputError();
  }

  function updateUploadButtonState() {
    const canSubmit = selectedFile !== null && permissionCheckbox.checked;
    uploadAnalyzeBtn.disabled = !canSubmit;
  }

  // --- Video Upload Form Submit ---
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      showVideoInputError("Please select a video file to analyze.");
      return;
    }

    if (!permissionCheckbox.checked) {
      showVideoInputError("You must confirm permission to analyze this video file.");
      permissionCheckbox.focus();
      return;
    }

    hideVideoInputError();
    const sampleInterval = parseFloat(sampleIntervalSelect.value) || 1.0;
    const minSequence = parseFloat(minSequenceSelect.value) || 3.0;

    await performVideoUploadAnalysis(selectedFile, sampleInterval, minSequence);
  });

  // --- Quick Demo Button Trigger ---
  runDemoBtn.addEventListener("click", async () => {
    hideVideoInputError();
    const sampleType = demoTypeSelect.value || "repeated";
    const sampleInterval = parseFloat(sampleIntervalSelect.value) || 1.0;
    const minSequence = parseFloat(minSequenceSelect.value) || 2.5;

    await performSampleDemoAnalysis(sampleType, sampleInterval, minSequence);
  });

  // --- Error View Dismiss ---
  if (errorDismissBtn) {
    errorDismissBtn.addEventListener("click", () => {
      hideAllStates();
      if (currentActiveMode === "youtube") {
        urlInput.focus();
      }
    });
  }

  // --- Reset Handlers ---
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      hideAllStates();
      urlInput.value = "";
      clearBtn.style.display = "none";
      urlInput.focus();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  if (resetVideoAnalysisBtn) {
    resetVideoAnalysisBtn.addEventListener("click", () => {
      hideAllStates();
      clearSelectedFile();
      permissionCheckbox.checked = false;
      updateUploadButtonState();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // --- API Handlers ---

  async function checkServerStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("Status check failed");
      const data = await res.json();

      statusBadge.classList.remove("checking", "online", "key-missing");
      if (data.api_key_configured) {
        statusBadge.classList.add("online");
        statusText.textContent = "Live YouTube API Connected & CV Ready";
        apiKeyBanner.style.display = "none";
      } else {
        statusBadge.classList.add("key-missing");
        statusText.textContent = "YouTube Key Needed (CV Ready)";
        apiKeyBanner.style.display = "block";
      }
    } catch (err) {
      statusBadge.classList.remove("checking");
      statusBadge.classList.add("key-missing");
      statusText.textContent = "Server Offline";
    }
  }

  async function performAnalysis(url) {
    setLoadingState(true, "youtube");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        handleApiError(response.status, data);
        return;
      }

      renderResults(data);
    } catch (err) {
      showErrorState(
        "Network Connection Error",
        "Unable to connect to the LoopLens server. Please ensure the backend is running."
      );
    } finally {
      setLoadingState(false, "youtube");
    }
  }

  async function performVideoUploadAnalysis(file, sampleInterval, minSequence) {
    setLoadingState(true, "video");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("sample_interval_sec", sampleInterval.toString());
      formData.append("min_sequence_sec", minSequence.toString());
      formData.append("permission_confirmed", "true");

      const response = await fetch("/api/analyze-video", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        const errorDetail = data.detail || data;
        const msg = errorDetail.message || "An error occurred during video frame repetition analysis.";
        showErrorState("Frame Analysis Failed", msg);
        return;
      }

      renderVideoResults(data);
    } catch (err) {
      showErrorState(
        "Upload or Analysis Error",
        "Could not complete frame repetition analysis. Please ensure the server is running and the video file is not corrupted."
      );
    } finally {
      setLoadingState(false, "video");
    }
  }

  async function performSampleDemoAnalysis(sampleType, sampleInterval, minSequence) {
    setLoadingState(true, "video");

    try {
      const formData = new FormData();
      formData.append("sample_type", sampleType);
      formData.append("sample_interval_sec", sampleInterval.toString());
      formData.append("min_sequence_sec", minSequence.toString());

      const response = await fetch("/api/analyze-sample-video", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        const errorDetail = data.detail || data;
        const msg = errorDetail.message || "An error occurred during sample video repetition analysis.";
        showErrorState("Sample Analysis Failed", msg);
        return;
      }

      renderVideoResults(data);
    } catch (err) {
      showErrorState(
        "Demonstration Error",
        "Failed to generate and analyze sample demonstration video. Please ensure OpenCV headless is installed."
      );
    } finally {
      setLoadingState(false, "video");
    }
  }

  function handleApiError(status, data) {
    const errorCode = data.error_code || "ERROR";
    const message = data.message || "An error occurred during video analysis.";

    if (errorCode === "API_KEY_REQUIRED") {
      apiKeyBanner.style.display = "block";
      showErrorState("YouTube API Key Not Configured", message);
    } else if (errorCode === "VIDEO_NOT_FOUND") {
      showErrorState("Video Not Found", message);
    } else if (errorCode === "QUOTA_EXCEEDED") {
      showErrorState("YouTube API Quota Reached", message);
    } else if (errorCode === "INVALID_URL") {
      showErrorState("Invalid Video Address", message);
    } else {
      showErrorState(`Analysis Request Failed (${status})`, message);
    }
  }

  // --- Render YouTube Results ---
  function renderResults(payload) {
    hideAllStates();
    const video = payload.video;
    const analysis = payload.analysis;

    videoThumb.src = video.thumbnail_url || "";
    videoThumb.alt = video.title || "Video Thumbnail";
    videoDurationBadge.textContent = video.duration_formatted || "0:00";
    videoTitle.textContent = video.title || "Untitled Video";
    videoChannel.textContent = video.channel_title || "Unknown Channel";
    videoViews.textContent = formatNumber(video.view_count);
    videoDurationText.textContent = video.duration_formatted || "--";
    videoComments.textContent = video.comment_count !== null ? formatNumber(video.comment_count) : "N/A";

    if (video.published_at) {
      const pubDate = new Date(video.published_at);
      videoPublishedDate.textContent = `Published ${pubDate.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })}`;
    } else {
      videoPublishedDate.textContent = "";
    }

    const resultState = analysis.result_state || "Footage analysis unavailable";
    const statusTitle = analysis.status_title || resultState;

    // Check if footage analysis succeeded with legitimate findings
    const hasLegitimateFootageResult = Boolean(
      analysis.direct_footage_analysis_performed &&
      analysis.visual_analysis &&
      (resultState === "Repeated footage detected" ||
       resultState === "No qualifying repetition detected" ||
       resultState === "Repeated sequence detected" ||
       resultState === "Inconclusive")
    );

    // Check if community comments yielded authentic flagged repetition
    const hasLegitimateCommunityResult = Boolean(
      !analysis.direct_footage_analysis_performed &&
      (analysis.repeat_reports_count > 0 || analysis.status === "REPEATED_FOOTAGE_REPORTED" || analysis.status === "POTENTIAL_ISSUES_NOTED")
    );

    // Completely remove/hide the error-result section when in error or unavailable state
    const isErrorOrUnavailableCard = 
      resultState === "Analysis unavailable" ||
      resultState === "Footage analysis unavailable" ||
      analysis.status === "ANALYSIS_UNAVAILABLE" ||
      analysis.status === "CLEAR" ||
      analysis.status === "COMMENTS_DISABLED" ||
      analysis.status === "NO_COMMENTS_AVAILABLE" ||
      statusTitle.toLowerCase().includes("unavailable") ||
      statusTitle.toLowerCase().includes("blocked");

    if (verdictCard) {
      if (isErrorOrUnavailableCard || (!hasLegitimateFootageResult && !hasLegitimateCommunityResult)) {
        // Completely remove this error section from the rendered website
        verdictCard.style.display = "none";
      } else {
        // Show legitimate scan results only
        verdictCard.style.display = "block";
        verdictCard.className = "verdict-card";
        const severity = analysis.severity || "neutral";
        verdictCard.classList.add(`verdict-${severity}`);
        verdictTag.textContent = getVerdictTag(resultState, severity, analysis.status);
        verdictTitle.textContent = (statusTitle && !statusTitle.toLowerCase().includes("unavailable"))
          ? statusTitle
          : (hasLegitimateFootageResult ? resultState : "Community Repetition Reported");
        
        // Ensure no HTTP 403 or API blocked descriptions are shown
        let cleanSummary = analysis.summary || "";
        if (cleanSummary.includes("Gemini API rejected") || cleanSummary.includes("HTTP 403")) {
          cleanSummary = cleanSummary.split("Direct AI")[0].trim();
        }
        verdictSummary.textContent = cleanSummary;
        if (verdictNotice) verdictNotice.textContent = analysis.notice || "";
        verdictIconBox.innerHTML = getVerdictIconSvg(severity);
      }
    }

    // 2b. Evidence-Based Pipeline Sources
    const pipeline = analysis.evidence_pipeline || {};
    if (pipeline.metadata && pipeMetaDesc) {
      pipeMetaDesc.textContent = pipeline.metadata.summary || "Evaluated video title, creator description, duration, and chapter markers.";
      if (pipeMetaStatus) {
        pipeMetaStatus.textContent = pipeline.metadata.evaluated ? "Evaluated" : "Unavailable";
        pipeMetaStatus.className = "pipeline-status-badge " + (pipeline.metadata.evaluated ? "status-badge-evaluated" : "status-badge-unperformed");
      }
    }
    if (pipeline.community_comments && pipeCommunityDesc) {
      pipeCommunityDesc.textContent = pipeline.community_comments.summary || "Evaluated public viewer comments retrieved via YouTube API v3.";
      if (pipeCommunityStatus) {
        const isEval = pipeline.community_comments.evaluated;
        pipeCommunityStatus.textContent = isEval ? "Evaluated" : (pipeline.community_comments.status === "COMMENTS_DISABLED" ? "Disabled" : "No Comments");
        pipeCommunityStatus.className = "pipeline-status-badge " + (isEval ? "status-badge-evaluated" : "status-badge-unperformed");
      }
    }
    if (pipeline.direct_footage && pipeFootageDesc) {
      const footage = pipeline.direct_footage;
      if (pipeFootageStatus) {
        if (footage.performed) {
          const modelTag = footage.model_used ? ` (${footage.model_used})` : "";
          pipeFootageStatus.textContent = `Completed${modelTag}`;
          pipeFootageStatus.className = "pipeline-status-badge status-badge-evaluated";
        } else {
          pipeFootageStatus.textContent = "Unavailable";
          pipeFootageStatus.className = "pipeline-status-badge status-badge-unperformed";
        }
      }
      if (footage.performed) {
        pipeFootageDesc.textContent = footage.summary || "Gemini multimodal video repetition analysis completed.";
      } else {
        pipeFootageDesc.textContent = "Direct visual footage analysis is unavailable for remote YouTube URLs via official APIs.";
      }
    }

    // 2c. Render Direct Visual Footage Analysis if performed
    if (analysis.direct_footage_analysis_performed && analysis.visual_analysis && ytVisualSection) {
      const vResult = analysis.visual_analysis;
      const vVerdict = vResult.verdict || {};
      const vMatches = vResult.matches || [];
      ytVisualSection.style.display = "block";
      if (ytMetricVideoTotalMatches) ytMetricVideoTotalMatches.textContent = vVerdict.total_candidate_matches ?? vMatches.length;
      if (ytMetricVideoDynamicMatches) ytMetricVideoDynamicMatches.textContent = vVerdict.dynamic_matches_count ?? 0;
      if (ytMetricVideoStaticMatches) ytMetricVideoStaticMatches.textContent = vVerdict.static_matches_count ?? 0;
      const totalRepSec = vMatches.filter(m => !m.is_static_scene).reduce((acc, m) => acc + (m.duration_sec || 0), 0);
      if (ytMetricVideoRepeatSec) ytMetricVideoRepeatSec.textContent = totalRepSec > 0 ? `${totalRepSec.toFixed(1)}s` : "0s";
      if (ytVideoMatchesList) {
        ytVideoMatchesList.innerHTML = "";
        if (vMatches.length === 0) {
          ytVideoMatchesList.innerHTML = `
            <div class="no-evidence-box">
              <div class="check-icon">✓</div>
              <p><strong>No qualifying repeated sequence detected in the analysed footage.</strong></p>
            </div>
          `;
        } else {
          vMatches.forEach((match, idx) => {
            const card = createVideoMatchCard(match, idx + 1);
            ytVideoMatchesList.appendChild(card);
          });
        }
      }
    } else if (ytVisualSection) {
      ytVisualSection.style.display = "none";
    }

    // 3. Metrics
    metricAnalyzed.textContent = analysis.total_analyzed ?? 0;
    metricFlagged.textContent = analysis.evidence_count ?? 0;
    metricRepeat.textContent = analysis.repeat_reports_count ?? 0;
    if (metricRecap) metricRecap.textContent = analysis.recap_reports_count ?? 0;
    metricIncomplete.textContent = analysis.incomplete_reports_count ?? 0;

    // 4. Timestamps
    const communityTimestamps = analysis.community_timestamps || [];
    const creatorChapters = (pipeline.metadata && pipeline.metadata.chapters) || [];

    if (communityTimestamps.length > 0 || creatorChapters.length > 0) {
      timestampsCard.style.display = "block";
      let tsHtml = "";
      creatorChapters.forEach((ch) => {
        tsHtml += `<span class="timestamp-chip chip-chapter" title="Creator Chapter">📑 ${escapeHtml(ch.timestamp)}: ${escapeHtml(ch.label)} (Creator Chapter)</span> `;
      });
      communityTimestamps.forEach((ts) => {
        tsHtml += `<span class="timestamp-chip" title="Mentioned in Viewer Comments">⏱️ ${escapeHtml(ts)} (Viewer Mention)</span> `;
      });
      timestampsList.innerHTML = tsHtml;
    } else {
      timestampsCard.style.display = "none";
      timestampsList.innerHTML = "";
    }

    evidenceCommentsList.innerHTML = "";
    const evidenceList = analysis.evidence_comments || [];

    if (evidenceList.length === 0) {
      evidenceCommentsList.innerHTML = `
        <div class="no-evidence-box">
          <div class="check-icon">✓</div>
          <p><strong>Zero repetition or cutoff reports detected in evaluated public comments.</strong></p>
          <p style="font-size: 0.82rem; margin-top: 0.4rem; color: var(--text-muted); line-height: 1.5;">
            <strong>Important:</strong> The absence of viewer comments cannot be used as evidence that a video has no loops.
            Direct visual footage analysis is unavailable through official YouTube APIs.
          </p>
        </div>
      `;
    } else {
      evidenceList.forEach((comment) => {
        const card = createCommentCard(comment);
        evidenceCommentsList.appendChild(card);
      });
    }

    resultsView.style.display = "flex";
    resultsView.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // --- Render Video Frame Analysis Results ---
  function renderVideoResults(payload) {
    hideAllStates();
    const info = payload.file_info || {};
    const verdict = payload.verdict || {};
    const matches = payload.matches || [];

    // 1. File Metadata
    fileDisplayName.textContent = info.filename || "Uploaded Video";
    fileDurationBadge.textContent = info.duration_formatted || "0:00";
    fileSpecsSummary.textContent = `${info.resolution || "Unknown"} • ${info.fps || "--"} fps • Local File Analysis`;
    fileDurationText.textContent = info.duration_formatted || "--";
    fileSampledCount.textContent = info.sampled_frames_count ?? "--";
    fileIntervalText.textContent = `${info.sample_interval_sec ?? 1.0}s`;
    fileTotalFrames.textContent = info.total_frames ? formatNumber(info.total_frames) : "--";

    // 2. Verdict Card
    videoVerdictCard.className = "verdict-card";
    const severity = verdict.severity || "neutral";
    videoVerdictCard.classList.add(`verdict-${severity}`);

    const resultState = verdict.result_state || "Repeated sequence detected";
    const subClass = verdict.sub_classification || "";
    videoVerdictTag.textContent = getVerdictTag(resultState, severity, verdict.status, subClass);
    videoVerdictTitle.textContent = verdict.status_title || resultState;
    videoVerdictSummary.textContent = verdict.summary || "";
    videoVerdictNotice.textContent = payload.prototype_disclaimer || "";
    videoVerdictIconBox.innerHTML = getVerdictIconSvg(severity);

    // 3. Metrics
    metricVideoTotalMatches.textContent = verdict.total_candidate_matches ?? matches.length;
    metricVideoDynamicMatches.textContent = verdict.dynamic_matches_count ?? 0;
    metricVideoStaticMatches.textContent = verdict.static_matches_count ?? 0;

    const totalRepeatSec = matches
      .filter((m) => !m.is_static_scene)
      .reduce((acc, m) => acc + (m.duration_sec || 0), 0);
    metricVideoRepeatSec.textContent = totalRepeatSec > 0 ? `${totalRepeatSec.toFixed(1)}s` : "0s";

    // 4. Matches List
    videoMatchesList.innerHTML = "";

    if (matches.length === 0) {
      videoMatchesList.innerHTML = `
        <div class="no-evidence-box">
          <div class="check-icon">✓</div>
          <p><strong>No qualifying repeated sequence detected in the analysed footage.</strong></p>
          <p style="font-size: 0.8rem; margin-top: 0.3rem;">
            Analyzed ${info.sampled_frames_count || 0} frames sampled at ${info.sample_interval_sec || 1.0}s intervals.
            No sequence exceeded the ${info.min_sequence_sec || 3.0}s repetition threshold.
          </p>
        </div>
      `;
    } else {
      matches.forEach((match, idx) => {
        const card = createVideoMatchCard(match, idx + 1);
        videoMatchesList.appendChild(card);
      });
    }

    videoResultsView.style.display = "flex";
    videoResultsView.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function createVideoMatchCard(match, index) {
    const el = document.createElement("article");
    el.className = "video-match-card";

    const orig = match.original_sequence || {};
    const rep = match.repeated_sequence || {};
    const thumbs = match.evidence_thumbnails || {};

    const confidenceClass = match.is_static_scene
      ? "confidence-static"
      : match.confidence === "High"
      ? "confidence-high"
      : "confidence-moderate";

    const classificationBadgeHtml = match.classification === "Possible loop-padding"
      ? `<span class="match-badge-padding">⚠️ Possible loop-padding</span>`
      : (!match.is_static_scene
        ? `<span class="match-badge-repeated">🔁 Repeated sequence</span>`
        : "");

    const staticWarningHtml = match.is_static_scene
      ? `
        <div class="match-static-alert" role="alert">
          <span>⚠️</span>
          <span><strong>Static Scene Flag:</strong> ${escapeHtml(match.static_scene_warning || "Minimal inter-frame motion detected. Likely a static slide or title card.")}</span>
        </div>
      `
      : "";

    const intentNoteHtml = `
      <div class="match-intent-note">
        <span>ℹ️</span>
        <span>${escapeHtml(match.classification_note || "Candidate repeated footage sequence.")} <em>Creator intent cannot be inferred from visual similarity alone.</em></span>
      </div>
    `;

    el.innerHTML = `
      <div class="match-header-row">
        <div class="match-title-group">
          <span class="match-number-badge">#${index}</span>
          <span class="match-time-range">${escapeHtml(orig.time_range || orig.start_formatted)} &rarr; ${escapeHtml(rep.time_range || rep.start_formatted)}</span>
        </div>
        <div class="match-badges-group">
          ${classificationBadgeHtml}
          <span class="match-duration-badge">⏱️ ${escapeHtml(match.duration_formatted || match.duration_sec + "s")}</span>
          <span class="match-score-badge">🔁 ${match.similarity_score}% Similar</span>
          <span class="match-confidence-badge ${confidenceClass}">Confidence: ${escapeHtml(match.confidence || "Moderate")}</span>
        </div>
      </div>

      ${staticWarningHtml}

      <div class="side-by-side-evidence">
        <div class="evidence-frame-col">
          <span class="evidence-frame-label">Original Sequence</span>
          ${thumbs.original_midpoint_b64 ? `
            <img 
              src="data:image/jpeg;base64,${thumbs.original_midpoint_b64}" 
              alt="Original footage keyframe at ${escapeHtml(thumbs.original_mid_time || orig.start_formatted)}" 
              class="evidence-thumb-img" 
              loading="lazy" 
            />
            <span class="evidence-time-label">⏱️ ${escapeHtml(thumbs.original_mid_time || orig.start_formatted)}</span>
          ` : `
            <div class="evidence-timeline-pill">
              <span class="timeline-icon">🎬</span>
              <span class="timeline-time">${escapeHtml(orig.time_range || orig.start_formatted)}</span>
            </div>
            <p class="evidence-desc-label">${escapeHtml(match.visual_description || "Original sequence segment")}</p>
          `}
        </div>

        <div class="similarity-meter-col">
          <span class="meter-arrow">⇄</span>
          <span class="sim-pill">${match.similarity_score}% Match</span>
        </div>

        <div class="evidence-frame-col">
          <span class="evidence-frame-label">Repeated Sequence</span>
          ${thumbs.repeated_midpoint_b64 ? `
            <img 
              src="data:image/jpeg;base64,${thumbs.repeated_midpoint_b64}" 
              alt="Candidate repeated footage keyframe at ${escapeHtml(thumbs.repeated_mid_time || rep.start_formatted)}" 
              class="evidence-thumb-img" 
              loading="lazy" 
            />
            <span class="evidence-time-label">⏱️ ${escapeHtml(thumbs.repeated_mid_time || rep.start_formatted)}</span>
          ` : `
            <div class="evidence-timeline-pill">
              <span class="timeline-icon">🔁</span>
              <span class="timeline-time">${escapeHtml(rep.time_range || rep.start_formatted)}</span>
            </div>
            <p class="evidence-desc-label">${escapeHtml(match.temporal_context || "Repeated sequence segment")}</p>
          `}
        </div>
      </div>

      ${intentNoteHtml}
    `;

    return el;
  }

  function createCommentCard(item) {
    const el = document.createElement("article");
    el.className = "comment-card";

    const tagsHtml = (item.categories || [])
      .map((cat) => {
        if (cat === "repeated_footage") return `<span class="evidence-tag tag-repeat">🔁 Repetition Mention</span>`;
        if (cat === "incomplete_video") return `<span class="evidence-tag tag-incomplete">✂️ Cut-off Mention</span>`;
        if (cat === "intentional_recap") return `<span class="evidence-tag tag-recap">📑 Recap / Montage Mention</span>`;
        if (cat === "ambient_loop") return `<span class="evidence-tag tag-recap">🔁 Extended Loop Mention</span>`;
        return `<span class="evidence-tag tag-filler">⏱️ Filler Mention</span>`;
      })
      .join(" ");

    const avatarHtml = item.author_avatar
      ? `<img src="${escapeHtml(item.author_avatar)}" alt="" class="comment-avatar" loading="lazy" />`
      : `<div class="comment-avatar-fallback">${escapeHtml((item.author || "A")[0].toUpperCase())}</div>`;

    el.innerHTML = `
      <div class="comment-header">
        <div class="comment-author-box">
          ${avatarHtml}
          <span class="comment-author-name">${escapeHtml(item.author || "Viewer")}</span>
        </div>
        <div class="comment-likes-badge">
          <span>👍</span>
          <span>${formatNumber(item.like_count)}</span>
        </div>
      </div>
      <p class="comment-text">${highlightTerms(escapeHtml(item.text))}</p>
      <div class="comment-tags-row">
        ${tagsHtml}
      </div>
    `;

    return el;
  }

  function highlightTerms(safeText) {
    const terms = [
      "repeat", "repeats", "repeating", "repeated",
      "loop", "loops", "looped", "looping",
      "cut off", "cuts off", "incomplete", "ends abruptly",
      "plays twice", "same footage", "same scene", "starts over",
      "recap", "recaps", "flashback", "flashbacks", "montage", "compilation",
      "previously on"
    ];
    let highlighted = safeText;
    terms.forEach((term) => {
      const regex = new RegExp(`\\b(${term})\\b`, "gi");
      highlighted = highlighted.replace(regex, `<mark class="highlight-keyword">$1</mark>`);
    });
    return highlighted;
  }

  function setLoadingState(isLoading, mode = "youtube") {
    const loadingTitle = document.querySelector(".loading-title");

    if (isLoading) {
      hideAllStates();
      loadingView.style.display = "block";

      if (mode === "youtube") {
        analyzeBtn.disabled = true;
        btnSpinner.style.display = "inline-block";
        btnIcon.style.display = "none";
        btnText.textContent = "Scanning...";
        if (loadingTitle) loadingTitle.textContent = "Scanning YouTube Video Data";

        const ytSteps = [
          "Connecting to YouTube Data API v3...",
          "Fetching video metadata and durations...",
          "Retrieving public viewer comment threads...",
          "Scanning comments for repetition & cut-off flags...",
          "Compiling authentic evidence report..."
        ];
        let stepIndex = 0;
        loadingStepText.textContent = ytSteps[0];
        loadingInterval = setInterval(() => {
          stepIndex = (stepIndex + 1) % ytSteps.length;
          loadingStepText.textContent = ytSteps[stepIndex];
        }, 1200);
      } else {
        uploadAnalyzeBtn.disabled = true;
        runDemoBtn.disabled = true;
        uploadBtnSpinner.style.display = "inline-block";
        uploadBtnIcon.style.display = "none";
        uploadBtnText.textContent = "Analyzing Frames...";
        if (loadingTitle) loadingTitle.textContent = "Analyzing Video Frame Repetition (CV Prototype)";

        const videoSteps = [
          "Uploading video file to server...",
          "Decoding video stream and sampling frames at interval...",
          "Computing 256-bit perceptual difference hashes (dHash)...",
          "Generating normalized luminance matrices and HSV color histograms...",
          "Tracing diagonal sequence runs to find repeated footage blocks...",
          "Applying inter-frame motion heuristics to filter static slides...",
          "Extracting visual keyframe evidence thumbnails..."
        ];
        let stepIndex = 0;
        loadingStepText.textContent = videoSteps[0];
        loadingInterval = setInterval(() => {
          stepIndex = (stepIndex + 1) % videoSteps.length;
          loadingStepText.textContent = videoSteps[stepIndex];
        }, 1100);
      }
    } else {
      if (loadingInterval) clearInterval(loadingInterval);
      loadingView.style.display = "none";

      if (mode === "youtube") {
        analyzeBtn.disabled = false;
        btnSpinner.style.display = "none";
        btnIcon.style.display = "inline-flex";
        btnText.textContent = "Analyze Video";
      } else {
        updateUploadButtonState();
        runDemoBtn.disabled = false;
        uploadBtnSpinner.style.display = "none";
        uploadBtnIcon.style.display = "inline-flex";
        uploadBtnText.textContent = "Analyze Video Frames";
      }
    }
  }

  function showErrorState(title, description) {
    hideAllStates();
    errorTitle.textContent = title;
    errorDesc.textContent = description;
    errorView.style.display = "block";
    errorView.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function hideAllStates() {
    loadingView.style.display = "none";
    errorView.style.display = "none";
    resultsView.style.display = "none";
    videoResultsView.style.display = "none";
    if (ytVisualSection) ytVisualSection.style.display = "none";
    if (verdictCard) verdictCard.style.display = "none";
  }

  function showInputError(msg) {
    inputError.textContent = msg;
    inputError.style.display = "block";
  }

  function hideInputError() {
    inputError.textContent = "";
    inputError.style.display = "none";
  }

  function showVideoInputError(msg) {
    videoInputError.textContent = msg;
    videoInputError.style.display = "block";
  }

  function hideVideoInputError() {
    videoInputError.textContent = "";
    videoInputError.style.display = "none";
  }

  // --- Utility Helpers ---

  function isValidYouTubeUrlOrId(input) {
    const cleaned = input.trim();
    if (/^[A-Za-z0-9_-]{11}$/.test(cleaned)) return true;
    return /(?:youtube\.com\/(?:watch|shorts|embed|v)|youtu\.be\/)/i.test(cleaned);
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function formatNumber(num) {
    if (num === null || num === undefined) return "0";
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1).replace(/\.0$/, "") + "M";
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1).replace(/\.0$/, "") + "K";
    }
    return num.toLocaleString();
  }

  function getVerdictTag(resultState, severity, status, subClassification) {
    if (resultState === "Repeated footage detected" || resultState === "Repeated sequence detected") {
      return subClassification === "Possible loop-padding"
        ? "POSSIBLE LOOP-PADDING"
        : "REPEATED FOOTAGE DETECTED";
    }
    if (resultState === "No qualifying repetition detected" || resultState === "No qualifying repeated sequence detected in the analysed footage") {
      return "NO QUALIFYING REPETITION";
    }
    if (resultState === "Inconclusive") {
      return "INCONCLUSIVE";
    }
    if (resultState === "Analysis unavailable" || resultState === "Footage analysis unavailable") {
      if (status === "REPEATED_FOOTAGE_REPORTED") {
        return "COMMUNITY REPETITION REPORTED";
      }
      if (status === "INCOMPLETE_VIDEO_REPORTED") {
        return "COMMUNITY INCOMPLETE VIDEO REPORTED";
      }
      if (status === "INTENTIONAL_LOOP_CONTENT" || status === "INTENTIONAL_RECAP_NOTED") {
        return "INTENTIONAL CONTENT NOTED";
      }
      return "ANALYSIS UNAVAILABLE";
    }
    if (severity === "safe") return "NO QUALIFYING REPEATED SEQUENCE";
    if (severity === "warning") return "REPEATED FOOTAGE ALERT";
    if (severity === "caution") return "STATIC SCENE / NOTICE";
    return "STATUS REPORT";
  }

  function getVerdictIconSvg(severity) {
    if (severity === "safe") {
      return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
    } else if (severity === "caution") {
      return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    } else if (severity === "warning") {
      return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    }
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
