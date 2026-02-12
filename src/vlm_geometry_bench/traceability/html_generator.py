"""HTML generator for static traceability reports.

Uses Jinja2 templates to generate index.html, summary.html, and test.html pages.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader


# =============================================================================
# CSS STYLESHEET
# =============================================================================

CSS_STYLESHEET = """
:root {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-card: #1f2937;
    --text-primary: #e5e7eb;
    --text-secondary: #9ca3af;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --border: #374151;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background-color: var(--bg-secondary);
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 8px;
}

header h1 {
    font-size: 1.8rem;
    margin-bottom: 5px;
}

header .subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.back-link {
    color: var(--accent);
    text-decoration: none;
    display: inline-block;
    margin-bottom: 15px;
}

.back-link:hover {
    text-decoration: underline;
}

/* Cards */
.card {
    background-color: var(--bg-card);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid var(--border);
}

.card h2 {
    font-size: 1.2rem;
    margin-bottom: 15px;
    color: var(--accent);
}

.card h3 {
    font-size: 1rem;
    margin-bottom: 10px;
    color: var(--text-secondary);
}

/* Grid layouts */
.grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}

.grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

@media (max-width: 768px) {
    .grid-2, .grid-3 {
        grid-template-columns: 1fr;
    }
}

/* Info lists */
.info-list {
    list-style: none;
}

.info-list li {
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
}

.info-list li:last-child {
    border-bottom: none;
}

.info-list .label {
    color: var(--text-secondary);
}

.info-list .value {
    font-weight: 500;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

th {
    background-color: var(--bg-secondary);
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.85rem;
    text-transform: uppercase;
}

tr:hover {
    background-color: rgba(59, 130, 246, 0.1);
}

/* Badges/Tags */
.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
}

.badge-success {
    background-color: rgba(16, 185, 129, 0.2);
    color: var(--success);
}

.badge-error {
    background-color: rgba(239, 68, 68, 0.2);
    color: var(--error);
}

.badge-warning {
    background-color: rgba(245, 158, 11, 0.2);
    color: var(--warning);
}

.badge-info {
    background-color: rgba(59, 130, 246, 0.2);
    color: var(--accent);
}

/* Metric values */
.metric-good {
    color: var(--success);
}

.metric-warning {
    color: var(--warning);
}

.metric-bad {
    color: var(--error);
}

/* Buttons */
.btn {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 4px;
    text-decoration: none;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}

.btn-primary {
    background-color: var(--accent);
    color: white;
}

.btn-primary:hover {
    background-color: var(--accent-hover);
}

.btn-secondary {
    background-color: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
}

.btn-secondary:hover {
    background-color: var(--border);
}

/* Filters */
.filters {
    display: flex;
    gap: 15px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.filter-group label {
    color: var(--text-secondary);
    font-size: 0.85rem;
}

.filter-group select {
    background-color: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.9rem;
}

/* Comment box */
.comment-box {
    background-color: var(--bg-secondary);
    border-left: 4px solid var(--accent);
    padding: 15px;
    margin: 15px 0;
    font-style: italic;
    color: var(--text-secondary);
}

/* Images */
.image-container {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

.image-box {
    flex: 1;
    min-width: 300px;
}

.image-box img {
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
}

.image-box .caption {
    text-align: center;
    margin-top: 8px;
    color: var(--text-secondary);
    font-size: 0.85rem;
}

/* Conversation history */
.conversation {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}

.turn {
    border-bottom: 1px solid var(--border);
}

.turn:last-child {
    border-bottom: none;
}

.turn-header {
    background-color: var(--bg-secondary);
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.turn-header .role {
    font-weight: 500;
}

.turn-header .role.user {
    color: var(--accent);
}

.turn-header .role.assistant {
    color: var(--success);
}

.turn-header .meta {
    color: var(--text-secondary);
    font-size: 0.8rem;
}

.turn-content {
    padding: 15px;
    white-space: pre-wrap;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 0.85rem;
    max-height: 300px;
    overflow-y: auto;
}

/* Positions list */
.positions-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}

.positions-box {
    background-color: var(--bg-secondary);
    border-radius: 8px;
    padding: 15px;
}

.positions-box h4 {
    margin-bottom: 10px;
    font-size: 0.9rem;
}

.positions-list {
    max-height: 200px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.8rem;
}

.positions-list div {
    padding: 2px 0;
}

/* Run list table specifics */
.run-comment {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-style: italic;
    margin-top: 5px;
}

.model-list {
    font-size: 0.85rem;
}
"""

# =============================================================================
# JAVASCRIPT
# =============================================================================

JAVASCRIPT = """
// Filter functionality
function applyFilters() {
    const modelFilter = document.getElementById('model-filter')?.value?.toLowerCase() || '';
    const backendFilter = document.getElementById('backend-filter')?.value?.toLowerCase() || '';
    const taskFilter = document.getElementById('task-filter')?.value?.toLowerCase() || '';
    const classFilter = document.getElementById('class-filter')?.value?.toLowerCase() || '';
    const statusFilter = document.getElementById('status-filter')?.value?.toLowerCase() || '';

    const rows = document.querySelectorAll('table tbody tr');
    rows.forEach(row => {
        const model = row.dataset.model?.toLowerCase() || '';
        const backend = row.dataset.backend?.toLowerCase() || '';
        const task = row.dataset.task?.toLowerCase() || '';
        const imgClass = row.dataset.class?.toLowerCase() || '';
        const status = row.dataset.status?.toLowerCase() || '';

        const matchModel = !modelFilter || model.includes(modelFilter);
        const matchBackend = !backendFilter || backend === backendFilter;
        const matchTask = !taskFilter || task === taskFilter;
        const matchClass = !classFilter || imgClass === classFilter;
        const matchStatus = !statusFilter || status === statusFilter;

        row.style.display = matchModel && matchBackend && matchTask && matchClass && matchStatus ? '' : 'none';
    });
}

// Model selector for multi-model comparison
function selectModel(modelName) {
    const tables = document.querySelectorAll('.model-tests-table');
    tables.forEach(table => {
        const tableModel = table.dataset.model;
        table.style.display = tableModel === modelName ? '' : 'none';
    });

    // Update selector highlight
    const selectors = document.querySelectorAll('.model-selector-btn');
    selectors.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.model === modelName);
    });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    // Add event listeners to filters
    const filters = document.querySelectorAll('.filter-select');
    filters.forEach(filter => {
        filter.addEventListener('change', applyFilters);
    });

    // Initialize model selector if present
    const firstModelBtn = document.querySelector('.model-selector-btn');
    if (firstModelBtn) {
        selectModel(firstModelBtn.dataset.model);
    }
});
"""

# =============================================================================
# JINJA2 TEMPLATES
# =============================================================================

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Geometry Bench - Test Runs</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>VLM Geometry Bench - Test Runs</h1>
            <p class="subtitle">Evaluation results for vision-language model geometric shape recognition</p>
        </header>

        <div class="filters">
            <div class="filter-group">
                <label>Backend:</label>
                <select id="backend-filter" class="filter-select">
                    <option value="">All</option>
                    {% for backend in backends %}
                    <option value="{{ backend }}">{{ backend }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="filter-group">
                <label>Task:</label>
                <select id="task-filter" class="filter-select">
                    <option value="">All</option>
                    {% for task in tasks %}
                    <option value="{{ task }}">{{ task }}</option>
                    {% endfor %}
                </select>
            </div>
        </div>

        <div class="card">
            <h2>Test Runs ({{ runs|length }} total)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Run ID</th>
                        <th>Models</th>
                        <th>Date</th>
                        <th>Duration</th>
                        <th>Tasks</th>
                        <th>Tests</th>
                        <th>Cost</th>
                        <th>Size</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {% for run in runs %}
                    <tr data-backend="{{ run.backends|join(',') }}" data-task="{{ run.tasks|join(',') }}">
                        <td>
                            <strong>{{ run.run_id }}</strong>
                            {% if run.comment %}
                            <div class="run-comment">"{{ run.comment }}"</div>
                            {% endif %}
                        </td>
                        <td class="model-list">{{ run.models|join(', ') }}</td>
                        <td>{{ run.timestamp[:16] | replace('T', ' ') if run.timestamp is string else run.timestamp.strftime('%Y-%m-%d %H:%M') }}</td>
                        <td>{{ format_duration(run.elapsed_seconds) }}</td>
                        <td>{{ run.tasks|join(', ') }}</td>
                        <td>{{ run.total_tests }}</td>
                        <td>{% if run.estimated_cost_usd %}${{ "%.2f"|format(run.estimated_cost_usd) }}{% if 'ollama' in run.backends %}*{% endif %}{% else %}free{% endif %}</td>
                        <td>{% if run.size_mb %}{{ "%.1f"|format(run.size_mb) }} MB{% else %}-{% endif %}</td>
                        <td><a href="{{ run.run_id }}/summary.html" class="btn btn-primary">View</a></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <p style="color: #8899aa; font-size: 0.85em; margin-top: 0.5em;">* Cost estimated using default token rates (not actual API pricing)</p>
        </div>
    </div>
    <script src="assets/script.js"></script>
</body>
</html>
"""

SUMMARY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ run.run_name }} - VLM Geometry Bench</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <div class="container">
        <a href="../index.html" class="back-link">← Back to All Runs</a>

        <header>
            <h1>Test Run: {{ run.run_id }}</h1>
            <p class="subtitle">{{ run.timestamp_start[:19] | replace('T', ' ') if run.timestamp_start is string else run.timestamp_start.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        </header>

        {% if run.comment %}
        <div class="comment-box">
            {{ run.comment }}
        </div>
        {% endif %}

        <div class="grid-2">
            <div class="card">
                <h2>Run Details</h2>
                <ul class="info-list">
                    <li><span class="label">Run Name</span><span class="value">{{ run.run_name }}</span></li>
                    <li><span class="label">Tasks</span><span class="value">{{ run.tasks|join(', ') }}</span></li>
                    <li><span class="label">Image Classes</span><span class="value">{{ run.image_classes|join(', ') }}</span></li>
                    <li><span class="label">Samples per Model</span><span class="value">{{ run.num_samples }}</span></li>
                    <li><span class="label">Models Evaluated</span><span class="value">{{ run.models|length }}</span></li>
                </ul>
            </div>

            <div class="card">
                <h2>Total Usage</h2>
                <ul class="info-list">
                    <li><span class="label">Elapsed Time</span><span class="value">{{ format_duration(total_elapsed) }}</span></li>
                    <li><span class="label">Total Tests</span><span class="value">{{ total_tests }}</span></li>
                    <li><span class="label">Total Tokens</span><span class="value">{{ "{:,}".format(total_tokens) }}</span></li>
                    <li><span class="label">Estimated Cost</span><span class="value">{% if total_cost %}${{ "%.2f"|format(total_cost) }}{% if 'ollama' in run.backends %}*{% endif %}{% else %}free{% endif %}</span></li>
                </ul>
            </div>
        </div>

        <div class="card">
            <h2>Model Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Backend</th>
                        <th>Host/URL</th>
                        <th>Success Rate</th>
                        {% for task in run.tasks %}
                        <th>{{ task }}</th>
                        {% endfor %}
                        <th>Tokens</th>
                        <th>Cost</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model_info in run.models %}
                    <tr>
                        <td><strong>{{ model_info.model }}</strong></td>
                        <td>{{ model_info.backend }}</td>
                        <td style="font-size: 0.8em;">{{ model_info.base_url or '-' }}</td>
                        <td class="{{ 'metric-good' if model_info.success_rate >= 90 else ('metric-warning' if model_info.success_rate >= 70 else 'metric-bad') }}">
                            {{ "%.1f"|format(model_info.success_rate) }}%
                        </td>
                        {% for task in run.tasks %}
                        <td>
                            {% set task_metrics = model_metrics.get(model_info.model, {}).get(task, {}) %}
                            {% if task == 'LOCATE' %}
                                {% set val = task_metrics.get('mean_detection_rate', 0) %}
                            {% elif task == 'COUNT' %}
                                {% set val = task_metrics.get('exact_match_rate', 0) %}
                            {% elif task == 'PATTERN' %}
                                {% set val = task_metrics.get('accuracy', 0) %}
                            {% else %}
                                {% set val = 0 %}
                            {% endif %}
                            <span class="{{ 'metric-good' if val >= 70 else ('metric-warning' if val >= 40 else 'metric-bad') }}">
                                {{ "%.1f"|format(val) }}%
                            </span>
                        </td>
                        {% endfor %}
                        <td>{{ "{:,}".format(model_info.input_tokens + model_info.output_tokens) }}</td>
                        <td>{% if model_info.estimated_cost_usd %}${{ "%.2f"|format(model_info.estimated_cost_usd) }}{% if model_info.backend == 'ollama' %}*{% endif %}{% else %}free{% endif %}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% if 'ollama' in run.backends %}
            <p style="color: #8899aa; font-size: 0.85em; margin-top: 0.5em;">* Cost estimated using default token rates (not actual API pricing)</p>
            {% endif %}
        </div>

        <div class="card">
            <h2>Individual Tests</h2>

            <div class="filters">
                <div class="filter-group">
                    <label>Model:</label>
                    <select id="model-filter" class="filter-select">
                        <option value="">All</option>
                        {% for model_info in run.models %}
                        <option value="{{ model_info.model }}">{{ model_info.model }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Class:</label>
                    <select id="class-filter" class="filter-select">
                        <option value="">All</option>
                        {% for cls in run.image_classes %}
                        <option value="{{ cls }}">{{ cls }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Status:</label>
                    <select id="status-filter" class="filter-select">
                        <option value="">All</option>
                        <option value="success">Success</option>
                        <option value="fail">Fail</option>
                    </select>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Sample ID</th>
                        <th>Model</th>
                        <th>Task</th>
                        <th>Class</th>
                        <th>Status</th>
                        <th>Primary Metric</th>
                        <th>Latency</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {% for test in tests %}
                    <tr data-model="{{ test.model }}" data-class="{{ test.image_class }}" data-status="{{ 'success' if test.success else 'fail' }}">
                        <td>{{ test.sample_id }}</td>
                        <td>{{ test.model }}</td>
                        <td>{{ test.task }}</td>
                        <td>{{ test.image_class }}</td>
                        <td>
                            {% if test.success %}
                            <span class="badge badge-success">OK</span>
                            {% else %}
                            <span class="badge badge-error">FAIL</span>
                            {% endif %}
                        </td>
                        <td>
                            {% if test.task == 'LOCATE' %}
                                Det: {{ "%.1f"|format(test.metrics.get('detection_rate', 0)) }}%
                            {% elif test.task == 'COUNT' %}
                                {% if test.metrics.get('exact_match') %}Exact{% else %}±{{ test.metrics.get('absolute_error', '?') }}{% endif %}
                            {% elif test.task == 'PATTERN' %}
                                {% if test.metrics.get('correct') %}Correct{% else %}Wrong{% endif %}
                            {% else %}
                                -
                            {% endif %}
                        </td>
                        <td>{{ test.total_latency_ms }}ms</td>
                        <td><a href="models/{{ safe_model_name(test.model) }}/tests/{{ test.sample_id }}_{{ test.task }}/test.html" class="btn btn-secondary">View</a></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    <script src="assets/script.js"></script>
</body>
</html>
"""

TEST_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ test.sample_id }} - {{ test.task }} - VLM Geometry Bench</title>
    <link rel="stylesheet" href="../../../assets/style.css">
</head>
<body>
    <div class="container">
        <a href="../../../summary.html" class="back-link">← Back to Run Summary</a>

        <header>
            <h1>{{ test.sample_id }} - {{ test.task }}</h1>
            <p class="subtitle">Model: {{ test.model }}</p>
        </header>

        <div class="grid-2">
            <div class="card">
                <h2>Test Details</h2>
                <ul class="info-list">
                    <li><span class="label">Sample ID</span><span class="value">{{ test.sample_id }}</span></li>
                    <li><span class="label">Task</span><span class="value">{{ test.task }}</span></li>
                    <li><span class="label">Image Class</span><span class="value">{{ test.image_class }}</span></li>
                    <li><span class="label">Model</span><span class="value">{{ test.model }}</span></li>
                    <li><span class="label">Status</span><span class="value">{% if test.success %}<span class="badge badge-success">SUCCESS</span>{% else %}<span class="badge badge-error">FAIL</span>{% endif %}</span></li>
                    <li><span class="label">Turns</span><span class="value">{{ test.num_turns }}</span></li>
                    <li><span class="label">Total Latency</span><span class="value">{{ test.total_latency_ms }}ms</span></li>
                    <li><span class="label">Tokens</span><span class="value">{{ test.total_input_tokens }} in / {{ test.total_output_tokens }} out</span></li>
                    {% if test.estimated_cost_usd %}
                    <li><span class="label">Cost</span><span class="value">${{ "%.4f"|format(test.estimated_cost_usd) }}{% if test.backend == 'ollama' %}*{% endif %}</span></li>
                    {% endif %}
                </ul>
            </div>

            <div class="card">
                <h2>Metrics</h2>
                <ul class="info-list">
                    {% for key, value in test.metrics.items() %}
                    <li>
                        <span class="label">{{ key }}</span>
                        <span class="value">
                            {% if value is number %}
                                {% if 'rate' in key or 'accuracy' in key or 'match' in key %}
                                    {{ "%.1f"|format(value) }}%
                                {% elif 'distance' in key %}
                                    {{ "%.4f"|format(value) }}
                                {% else %}
                                    {{ value }}
                                {% endif %}
                            {% elif value is sameas true %}
                                <span class="metric-good">Yes</span>
                            {% elif value is sameas false %}
                                <span class="metric-bad">No</span>
                            {% else %}
                                {{ value }}
                            {% endif %}
                        </span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        {% if other_models %}
        <div class="card">
            <h2>Compare Models on This Sample</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Status</th>
                        {% if test.task == 'LOCATE' %}
                        <th>Detection Rate</th>
                        <th>False Positive Rate</th>
                        {% elif test.task == 'COUNT' %}
                        <th>Exact Match</th>
                        <th>Absolute Error</th>
                        {% elif test.task == 'PATTERN' %}
                        <th>Correct</th>
                        <th>Predicted</th>
                        {% endif %}
                        <th>Latency</th>
                        <th>Cost</th>
                    </tr>
                </thead>
                <tbody>
                    {% for other in other_models %}
                    <tr>
                        <td>{% if other.model == test.model %}<strong>{{ other.model }}</strong>{% else %}{{ other.model }}{% endif %}</td>
                        <td>{% if other.success %}<span class="badge badge-success">OK</span>{% else %}<span class="badge badge-error">FAIL</span>{% endif %}</td>
                        {% if test.task == 'LOCATE' %}
                        <td>{{ "%.1f"|format(other.metrics.get('detection_rate', 0)) }}%</td>
                        <td>{{ "%.1f"|format(other.metrics.get('false_positive_rate', 0)) }}%</td>
                        {% elif test.task == 'COUNT' %}
                        <td>{% if other.metrics.get('exact_match') %}Yes{% else %}No{% endif %}</td>
                        <td>{{ other.metrics.get('absolute_error', '-') }}</td>
                        {% elif test.task == 'PATTERN' %}
                        <td>{% if other.metrics.get('correct') %}Yes{% else %}No{% endif %}</td>
                        <td>{{ other.metrics.get('predicted', '-') }}</td>
                        {% endif %}
                        <td>{{ other.total_latency_ms }}ms</td>
                        <td>{% if other.estimated_cost_usd %}${{ "%.4f"|format(other.estimated_cost_usd) }}{% if test.backend == 'ollama' %}*{% endif %}{% else %}free{% endif %}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="card">
            <h2>Images</h2>
            <div class="image-container">
                <div class="image-box">
                    <img src="original.png" alt="Original Image">
                    <p class="caption">Original Image</p>
                </div>
                <div class="image-box">
                    <img src="annotated.png" alt="Annotated Result">
                    <p class="caption">Annotated Result</p>
                </div>
            </div>
        </div>

        {% if conversation and conversation.turns %}
        <div class="card">
            <h2>Conversation History ({{ conversation.turns|length // 2 }} turns)</h2>
            <div class="conversation">
                {% for turn in conversation.turns %}
                <div class="turn">
                    <div class="turn-header">
                        <span class="role {{ turn.role }}">{{ turn.role|upper }}{% if turn.image_attached %} (with image){% endif %}</span>
                        <span class="meta">
                            {% if turn.latency_ms %}{{ turn.latency_ms }}ms{% endif %}
                            {% if turn.output_tokens %}, {{ turn.output_tokens }} tokens{% endif %}
                        </span>
                    </div>
                    <div class="turn-content">{{ turn.content }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        {% if test.task == 'LOCATE' and (test.ground_truth.positions or test.prediction.positions) %}
        <div class="card">
            <h2>Position Details</h2>
            <div class="positions-grid">
                <div class="positions-box">
                    <h4>Ground Truth ({{ test.ground_truth.positions|length }} positions)</h4>
                    <div class="positions-list">
                        {% for pos in test.ground_truth.positions %}
                        <div>({{ "%.3f"|format(pos[0]) }}, {{ "%.3f"|format(pos[1]) }})</div>
                        {% endfor %}
                    </div>
                </div>
                <div class="positions-box">
                    <h4>Predicted ({{ test.prediction.positions|length }} positions)</h4>
                    <div class="positions-list">
                        {% for pos in test.prediction.positions %}
                        <div>({{ "%.3f"|format(pos[0]) }}, {{ "%.3f"|format(pos[1]) }})</div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
    </div>
    <script src="../../../assets/script.js"></script>
</body>
</html>
"""


# =============================================================================
# HTML GENERATOR CLASS
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}m"


def safe_model_name(model: str) -> str:
    """Convert model name to safe folder name."""
    return model.replace("/", "_").replace(":", "-").replace(" ", "_")


class HTMLGenerator:
    """Generates static HTML pages for traceability reports."""

    def __init__(self):
        """Initialize HTML generator with Jinja2 environment."""
        self.env = Environment(loader=BaseLoader())

        # Register custom filters
        self.env.filters["format_duration"] = format_duration

        # Register global functions
        self.env.globals["format_duration"] = format_duration
        self.env.globals["safe_model_name"] = safe_model_name

        # Compile templates
        self.index_template = self.env.from_string(INDEX_TEMPLATE)
        self.summary_template = self.env.from_string(SUMMARY_TEMPLATE)
        self.test_template = self.env.from_string(TEST_TEMPLATE)

    def generate_index_html(
        self,
        runs: List[Dict[str, Any]],
    ) -> str:
        """Generate the main index.html page.

        Args:
            runs: List of run entries (as dicts or RunIndexEntry objects)

        Returns:
            HTML string
        """
        # Convert Pydantic models to dicts if needed
        runs_data = []
        for run in runs:
            if hasattr(run, "model_dump"):
                runs_data.append(run.model_dump())
            else:
                runs_data.append(run)

        # Collect unique backends and tasks
        backends = set()
        tasks = set()
        for run in runs_data:
            backends.update(run.get("backends", []))
            tasks.update(run.get("tasks", []))

        return self.index_template.render(
            runs=runs_data,
            backends=sorted(backends),
            tasks=sorted(tasks),
        )

    def generate_summary_html(
        self,
        run: Dict[str, Any],
        tests: List[Dict[str, Any]],
        model_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> str:
        """Generate the summary.html page for a run.

        Args:
            run: RunMetadata as dict
            tests: List of TestMetadata as dicts
            model_metrics: Nested dict of model -> task -> aggregated metrics

        Returns:
            HTML string
        """
        # Calculate totals
        total_elapsed = sum(m.get("elapsed_seconds", 0) for m in run.get("models", []))
        total_tests = sum(m.get("total_tests", 0) for m in run.get("models", []))
        total_tokens = sum(
            m.get("input_tokens", 0) + m.get("output_tokens", 0)
            for m in run.get("models", [])
        )
        total_cost = sum(
            m.get("estimated_cost_usd") or 0
            for m in run.get("models", [])
        )

        return self.summary_template.render(
            run=run,
            tests=tests,
            model_metrics=model_metrics,
            total_elapsed=total_elapsed,
            total_tests=total_tests,
            total_tokens=total_tokens,
            total_cost=total_cost if total_cost > 0 else None,
        )

    def generate_test_html(
        self,
        test: Dict[str, Any],
        conversation: Optional[Dict[str, Any]] = None,
        other_models: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate the test.html page for an individual test.

        Args:
            test: TestMetadata as dict
            conversation: ConversationHistory as dict (optional)
            other_models: List of TestMetadata for same sample from other models (optional)

        Returns:
            HTML string
        """
        return self.test_template.render(
            test=test,
            conversation=conversation,
            other_models=other_models,
        )

    def get_css(self) -> str:
        """Get the CSS stylesheet content.

        Returns:
            CSS string
        """
        return CSS_STYLESHEET

    def get_js(self) -> str:
        """Get the JavaScript content.

        Returns:
            JavaScript string
        """
        return JAVASCRIPT

    def save_assets(self, assets_dir: Path) -> None:
        """Save CSS and JS assets to a directory.

        Args:
            assets_dir: Path to assets directory
        """
        assets_dir.mkdir(parents=True, exist_ok=True)

        css_path = assets_dir / "style.css"
        with open(css_path, "w", encoding="utf-8") as f:
            f.write(CSS_STYLESHEET)

        js_path = assets_dir / "script.js"
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(JAVASCRIPT)
