// 이메일 데이터 캐시
let lastEmailData = { Work: [], Personal: [], Ad: [] };
let categoryChart = null;
let relabels = [];

function renderCategoryChart(workCount, personalCount, adCount) {
    const ctx = document.getElementById('categoryChart').getContext('2d');
    const data = {
        labels: ['업무', '개인', '광고'],
        datasets: [{
            data: [workCount, personalCount, adCount],
            backgroundColor: [
                '#4e79a7', // 업무
                '#f28e2b', // 개인
                '#e15759'  // 광고
            ],
            borderWidth: 2
        }]
    };
    const options = {
        responsive: true,
        cutout: '70%',
        plugins: {
            legend: {
                display: true,
                position: 'bottom',
                labels: { font: { size: 14 } }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const label = context.label || '';
                        const value = context.parsed;
                        return `${label}: ${value}개`;
                    }
                }
            }
        }
    };
    if (categoryChart) {
        categoryChart.data = data;
        categoryChart.update();
    } else {
        categoryChart = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: options
        });
    }
}

function makeCategoryDropdown(selected) {
    return `<select class="relabel-category">
        <option value="Work" ${selected==='Work'?'selected':''}>업무</option>
        <option value="Personal" ${selected==='Personal'?'selected':''}>개인</option>
        <option value="Ad" ${selected==='Ad'?'selected':''}>광고</option>
    </select>`;
}

function renderEmailList(category, data) {
    const listId = {
        Work: 'work-list',
        Personal: 'personal-list',
        Ad: 'ad-list'
    }[category];
    const ul = document.getElementById(listId);
    ul.innerHTML = data.map((e, i) =>
        `<li style='display:flex;align-items:center;gap:8px;'>
            <div style='flex:1;' ondblclick='window.showEmailModal && window.showEmailModal("${category}", ${i})'>
                <strong>${e.subject}</strong><br><small>${e.sender} | ${e.date}</small>
            </div>
            <div>${makeCategoryDropdown(category)} <button class='relabel-btn' data-category='${category}' data-index='${i}'>재분류</button></div>
        </li>`
    ).join('');
}

function updateAllEmailLists(data) {
    renderEmailList('Work', data.Work);
    renderEmailList('Personal', data.Personal);
    renderEmailList('Ad', data.Ad);
    // 재분류 버튼 이벤트 바인딩
    document.querySelectorAll('.relabel-btn').forEach(btn => {
        btn.onclick = function() {
            const category = btn.getAttribute('data-category');
            const index = parseInt(btn.getAttribute('data-index'));
            const select = btn.parentElement.querySelector('.relabel-category');
            const newCategory = select.value;
            if (newCategory === category) {
                alert('다른 카테고리로 선택해 주세요.');
                return;
            }
            // 메일 정보 추출
            const email = lastEmailData[category][index];
            relabels.push({
                subject: email.subject,
                body: email.body,
                sender: email.sender,
                date: email.date,
                old_category: category,
                new_category: newCategory
            });
            // lastEmailData에서 이동
            lastEmailData[category].splice(index, 1);
            lastEmailData[newCategory].push(email);
            // UI 즉시 반영
            updateAllEmailLists(lastEmailData);
            // 카테고리별 개수로 도넛 차트도 즉시 갱신
            const workCount = lastEmailData.Work.length;
            const personalCount = lastEmailData.Personal.length;
            const adCount = lastEmailData.Ad.length;
            renderCategoryChart(workCount, personalCount, adCount);
            alert('재분류 내역이 임시 저장되었고, 메일이 새 카테고리로 이동했습니다.\n"재분류 내역 저장" 버튼을 눌러 서버에 저장하세요.');
            document.getElementById('save-relabels-btn').style.display = 'inline-block';
        };
    });
}

document.getElementById('classify-btn').addEventListener('click', async () => {
    const loadingDiv = document.getElementById('loading');
    const classifyBtn = document.getElementById('classify-btn');
    const maxResults = document.getElementById('max-results').value || 10;
    const unreadOnly = document.getElementById('unread-only').checked;
    const searchQuery = document.getElementById('search-query').value.trim();
    
    try {
        // 로딩 상태 표시
        loadingDiv.style.display = 'block';
        classifyBtn.disabled = true;
        classifyBtn.textContent = '분류 중...';
        
        // 쿼리 파라미터 조합
        const params = new URLSearchParams();
        params.append('max_results', maxResults);
        if (unreadOnly) params.append('unread', 'true');
        if (searchQuery) params.append('query', searchQuery);

        // 백엔드 API 호출
        const response = await fetch(`http://localhost:8000/gmail/classify_all?${params.toString()}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        lastEmailData = data; // 캐시에 저장

        // 카테고리별 개수 계산 및 차트 렌더링
        const workCount = data.Work.length;
        const personalCount = data.Personal.length;
        const adCount = data.Ad.length;
        renderCategoryChart(workCount, personalCount, adCount);

        // 카테고리별로 결과 표시 (제목/발신자/날짜만)
        updateAllEmailLists(data);

    } catch (error) {
        console.error('이메일 분류 오류:', error);
        alert('이메일 분류 중 오류가 발생했습니다: ' + error.message);
    } finally {
        // 로딩 상태 해제
        loadingDiv.style.display = 'none';
        classifyBtn.disabled = false;
        classifyBtn.textContent = '이메일 분류 실행';
    }
});

document.getElementById('save-relabels-btn').onclick = async function() {
    if (relabels.length === 0) {
        alert('재분류 내역이 없습니다.');
        return;
    }
    try {
        const response = await fetch('http://localhost:8000/gmail/save_relabels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ relabels })
        });
        if (!response.ok) throw new Error('서버 오류');
        alert('재분류 내역이 서버에 저장되었습니다!');
        relabels = [];
        document.getElementById('save-relabels-btn').style.display = 'none';
    } catch (e) {
        alert('저장 실패: ' + e.message);
    }
};

// 모달에서 본문 안전하게 렌더링
window.showEmailModal = function(category, idx) {
    const email = lastEmailData[category][idx];
    document.getElementById('modal-subject').textContent = email.subject;
    document.getElementById('modal-meta').textContent = `${email.sender || ''} | ${email.date || ''}`;
    // HTML 본문 안전하게 렌더링 (이미지 등 포함)
    document.getElementById('modal-body').innerHTML = email.body || '(본문 없음)';
    document.getElementById('email-modal').style.display = 'flex';
}; 