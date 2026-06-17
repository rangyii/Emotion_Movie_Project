let currentResults = [];

function checkLimit(checkbox) {
    const checkboxes = document.querySelectorAll('input[name="emotion"]:checked');
    const submitBtn = document.getElementById('submitBtn');

    if (checkboxes.length > 3) {
        alert("최대 3개까지만 선택할 수 있습니다.");
        checkbox.checked = false;
        return;
    }

    if (checkboxes.length === 0) {
        submitBtn.disabled = true;
        submitBtn.innerText = "감정을 선택해주세요";
        submitBtn.style.backgroundColor = "#555";
        submitBtn.style.color = "#888";
    } else {
        submitBtn.disabled = false;
        submitBtn.innerText = `${checkboxes.length}개 감정으로 추천받기`;
        submitBtn.style.backgroundColor = "#fff";
        submitBtn.style.color = "#141414";
    }
}

async function submitEmotions() {
    const checkboxes = document.querySelectorAll('input[name="emotion"]:checked');
    const selectedEmotions = Array.from(checkboxes).map(cb => cb.value);

    if (selectedEmotions.length === 0) return;

    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    const list = document.getElementById('recommendationList');

    list.innerHTML = '';
    resultSection.classList.add('hidden');
    loading.classList.remove('hidden');
    document.getElementById('loadingText').innerText = "복합적인 감정을 분석하여 영화를 매칭 중입니다...";

    try {
        const response = await fetch('/recommend_by_emotion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emotions: selectedEmotions })
        });

        const data = await response.json();
        loading.classList.add('hidden');

        if (data.status === 'success') {
            currentResults = data.results;
            resultSection.classList.remove('hidden');
            
            data.results.forEach((movie, index) => {
                const card = document.createElement('div');
                card.className = 'movie-card';
                card.onclick = () => openModal(index);
                
                let badgeColor = '#28a745'; 
                if (movie.score >= 95) badgeColor = '#e50914';
                else if (movie.score >= 90) badgeColor = '#ff9900';

                card.innerHTML = `
                    <div class="poster-wrapper">
                        <img src="${movie.poster}" class="poster-img">
                        <div class="score-badge" style="background-color:${badgeColor};">
                            ${movie.score}%
                        </div>
                    </div>
                    <div class="card-content">
                        <div class="movie-title">${movie.title}</div>
                        <div style="font-size:12px; color:#777; margin-bottom:8px;">${movie.year} | ${movie.genre.split(',')[0]}</div>
                        <div style="font-size:13px; color:#ccc; font-style:italic;">"${movie.reason}"</div>
                    </div>
                `;
                list.appendChild(card);
            });
        } else {
            alert(data.message || "오류가 발생했습니다.");
        }

    } catch (error) {
        console.error(error);
        loading.classList.add('hidden');
        alert("서버 연결 실패");
    }
}

function openModal(index) {
    const movie = currentResults[index];
    const modal = document.getElementById('movieModal');
    modal.style.display = "block";
    
    document.getElementById('modalPoster').src = movie.poster;
    document.getElementById('modalTitle').innerText = movie.title;
    document.getElementById('modalMeta').innerText = `${movie.year} | ${movie.genre}`;
    document.getElementById('modalPlot').innerText = movie.overview || "줄거리 정보가 없습니다.";
    
    document.getElementById('modalReason').innerHTML = `<span style="color:#fff;">${movie.reason}</span>`;
    document.getElementById('modalFood').innerText = movie.food || "팝콘";
}

function closeModal() {
    document.getElementById('movieModal').style.display = "none";
}

window.onclick = function(e) {
    if(e.target == document.getElementById('movieModal')) closeModal();
}