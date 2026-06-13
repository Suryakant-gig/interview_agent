// ======================================================
// PAGE NAVIGATION
// ======================================================

const titles = {
    dashboard: 'Dashboard',
    upload: 'Upload PDF',
    question: 'Generate Question',
    evaluate: 'Evaluate Answer'
};

function nav(page, el) {

    // remove active class from sidebar items
    document
        .querySelectorAll('.nav-item')
        .forEach(n => n.classList.remove('active'));

    // activate clicked item
    el.classList.add('active');

    // hide all pages
    document
        .querySelectorAll('.page')
        .forEach(p => p.classList.remove('active'));

    // show selected page
    document
        .getElementById('page-' + page)
        .classList.add('active');

    // update title
    document
        .getElementById('page-title')
        .textContent = titles[page];
}


// ======================================================
// MOCK PDF UPLOAD
// ======================================================

function simulateUpload() {

    const list = document.getElementById('file-list');

    const name = 'deep_learning_notes.pdf';

    const pages = Math.floor(Math.random() * 20) + 10;

    const chunks = pages * Math.floor(Math.random() * 8 + 6);

    list.innerHTML = `
        <div class="file-item">

            <i class="ti ti-file-type-pdf"></i>

            <div class="file-meta">

                <div class="file-name">
                    ${name}
                </div>

                <div class="file-size">
                    ${(Math.random() * 4 + 1).toFixed(1)} MB
                </div>

                <div class="progress-track">
                    <div class="progress-fill"
                         id="pbar"
                         style="width:0%">
                    </div>
                </div>

            </div>

        </div>
    `;

    let progress = 0;

    const interval = setInterval(() => {

        progress += Math.random() * 25;

        if (progress >= 100) {

            progress = 100;

            clearInterval(interval);

            showUploadResult(pages, chunks);
        }

        document.getElementById('pbar')
            .style.width = Math.round(progress) + '%';

    }, 200);
}


// ======================================================
// SHOW PDF PROCESSING RESULT
// ======================================================

function showUploadResult(pages, chunks) {

    const result = document.getElementById('upload-result');

    result.style.display = 'block';

    document.getElementById('pages-loaded')
        .textContent = pages;

    document.getElementById('chunks-created')
        .textContent = chunks;
}


// ======================================================
// SAMPLE QUESTIONS
// ======================================================

const sampleQuestions = {

    easy:
        'What is a convolutional neural network (CNN) and what types of problems is it primarily used for?',

    medium:
        'Explain how the spatial feature extraction process works in a CNN, including convolutional layers, pooling layers, and activation functions.',

    hard:
        'Compare CNNs and Vision Transformers (ViTs) for image recognition tasks and explain when each performs better.'
};


// ======================================================
// GENERATE QUESTION
// ======================================================

function generateQuestion() {

    const topic = document
        .getElementById('topic-query')
        .value
        .trim();

    const difficulty = document
        .getElementById('difficulty')
        .value;

    const result = document.getElementById('question-result');

    result.style.display = 'block';

    document.getElementById('q-difficulty-badge')
        .textContent =
            difficulty.charAt(0).toUpperCase()
            + difficulty.slice(1);

    document.getElementById('generated-question-text')
        .textContent = 'Generating...';

    setTimeout(() => {

        const text = topic
            ? `Given the topic "${topic}", ${sampleQuestions[difficulty].toLowerCase()}`
            : sampleQuestions[difficulty];

        document.getElementById('generated-question-text')
            .textContent = text;

    }, 900);
}


// ======================================================
// COPY QUESTION
// ======================================================

let lastQuestion = '';

function copyQuestion() {

    const text = document
        .getElementById('generated-question-text')
        .textContent;

    navigator.clipboard.writeText(text);

    lastQuestion = text;
}


// ======================================================
// SEND QUESTION TO EVALUATION PAGE
// ======================================================

function sendToEval() {

    const text = document
        .getElementById('generated-question-text')
        .textContent;

    document.getElementById('eval-question')
        .value = text;

    nav(
        'evaluate',
        document.querySelectorAll('.nav-item')[3]
    );
}


// ======================================================
// RUBRIC TAGS
// ======================================================

function addTag() {

    const input = document.getElementById('rubric-input');

    const value = input.value.trim();

    if (!value) return;

    const tag = document.createElement('div');

    tag.className = 'rubric-tag';

    tag.innerHTML = `
        ${value}
        <span class="del"
              onclick="removeTag(this)">
              ×
        </span>
    `;

    document.getElementById('rubric-list')
        .appendChild(tag);

    input.value = '';
}


function removeTag(el) {

    el.parentElement.remove();
}


// ======================================================
// MOCK EVALUATION RESULT
// ======================================================

const mockResult = {

    overall_score: 8,

    rubric_scores: {
        'Concept clarity': 8,
        'Key points covered': 7,
        'Example or explanation': 6
    },

    strengths: [
        'Correctly identifies convolution layers',
        'Mentions deep learning context'
    ],

    weaknesses: [
        'Does not explain pooling',
        'Missing activation functions'
    ],

    missing_concepts: [
        'Pooling layers',
        'ReLU activation',
        'Flatten layer'
    ],

    improvement:
        'Add pooling and ReLU explanations and describe how features connect to the classifier head.'
};


// ======================================================
// RUN EVALUATION
// ======================================================

function runEvaluation() {

    const result = document.getElementById('eval-result');

    result.style.display = 'none';

    setTimeout(() => {

        const r = mockResult;

        result.style.display = 'block';

        // overall score
        document.getElementById('overall-score-num')
            .textContent = r.overall_score;

        document.getElementById('overall-bar')
            .style.width = (r.overall_score * 10) + '%';

        // rubric scores
        const rubricBox = document.getElementById('rubric-scores');

        rubricBox.innerHTML = '';

        Object.entries(r.rubric_scores)
            .forEach(([key, value]) => {

                rubricBox.innerHTML += `
                    <div class="score-bar-row">

                        <span class="score-bar-label">
                            ${key}
                        </span>

                        <div class="score-bar-track">
                            <div class="score-bar-fill"
                                 style="width:${value * 10}%">
                            </div>
                        </div>

                        <span class="score-bar-val">
                            ${value}
                        </span>

                    </div>
                `;
            });

        // strengths
        document.getElementById('strengths-list')
            .innerHTML = r.strengths.map(item => `
                <div class="list-item">
                    <i class="ti ti-circle-check"></i>
                    <span>${item}</span>
                </div>
            `).join('');

        // weaknesses
        document.getElementById('weaknesses-list')
            .innerHTML = r.weaknesses.map(item => `
                <div class="list-item">
                    <i class="ti ti-alert-circle"></i>
                    <span>${item}</span>
                </div>
            `).join('');

        // missing concepts
        document.getElementById('missing-list')
            .innerHTML = r.missing_concepts.map(item => `
                <div class="list-item">
                    <i class="ti ti-minus"></i>
                    <span>${item}</span>
                </div>
            `).join('');

        // improvement text
        document.getElementById('improvement-text')
            .textContent = r.improvement;

    }, 1200);
}