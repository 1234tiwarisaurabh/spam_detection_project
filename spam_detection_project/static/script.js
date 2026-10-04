// ---------- Floating particles background ----------
const particleContainer = document.getElementById('particles');
for (let i = 0; i < 30; i++) {
  const p = document.createElement('div');
  p.className = 'particle';
  p.style.left = Math.random() * 100 + 'vw';
  p.style.animationDuration = (8 + Math.random() * 10) + 's';
  p.style.animationDelay = (Math.random() * 10) + 's';
  p.style.opacity = Math.random() * 0.6 + 0.2;
  particleContainer.appendChild(p);
}

// ---------- Elements ----------
const messageInput = document.getElementById('messageInput');
const checkBtn = document.getElementById('checkBtn');
const clearBtn = document.getElementById('clearBtn');
const resultBox = document.getElementById('result');
const resultIcon = document.getElementById('resultIcon');
const resultLabel = document.getElementById('resultLabel');
const resultSub = document.getElementById('resultSub');
const spamBar = document.getElementById('spamBar');
const hamBar = document.getElementById('hamBar');
const spamPct = document.getElementById('spamPct');
const hamPct = document.getElementById('hamPct');
const errorBox = document.getElementById('errorBox');
const btnText = document.querySelector('.btn-text');
const spinner = document.querySelector('.spinner');

// ---------- Example chips ----------
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    messageInput.value = chip.dataset.text;
    messageInput.focus();
  });
});

clearBtn.addEventListener('click', () => {
  messageInput.value = '';
  resultBox.classList.add('hidden');
  errorBox.classList.add('hidden');
  messageInput.focus();
});

async function analyzeMessage() {
  const text = messageInput.value.trim();
  errorBox.classList.add('hidden');

  if (!text) {
    errorBox.textContent = 'Please enter a message to analyze.';
    errorBox.classList.remove('hidden');
    return;
  }

  // loading state
  checkBtn.disabled = true;
  btnText.textContent = 'Analyzing...';
  spinner.classList.remove('hidden');
  resultBox.classList.add('hidden');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Something went wrong');
    }

    showResult(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove('hidden');
  } finally {
    checkBtn.disabled = false;
    btnText.textContent = 'Analyze Message';
    spinner.classList.add('hidden');
  }
}

function showResult(data) {
  const isSpam = data.label === 'spam';

  resultIcon.className = 'result-icon ' + (isSpam ? 'spam' : 'ham');
  resultIcon.textContent = isSpam ? '🚫' : '✅';

  resultLabel.className = isSpam ? 'spam' : 'ham';
  resultLabel.textContent = isSpam ? 'Spam Detected!' : 'Looks Safe (Ham)';

  resultSub.textContent = `Model confidence: ${data.confidence}%`;

  resultBox.classList.remove('hidden');

  // animate bars (reset then set, so transition triggers)
  spamBar.style.width = '0%';
  hamBar.style.width = '0%';
  spamPct.textContent = '0%';
  hamPct.textContent = '0%';

  requestAnimationFrame(() => {
    setTimeout(() => {
      spamBar.style.width = data.spam_probability + '%';
      hamBar.style.width = data.ham_probability + '%';
      spamPct.textContent = data.spam_probability + '%';
      hamPct.textContent = data.ham_probability + '%';
    }, 50);
  });
}

checkBtn.addEventListener('click', analyzeMessage);
messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    analyzeMessage();
  }
});
