// main.js - Frontend JavaScript untuk Sistem Indeks Buku

function showLoading() {
    var title = document.getElementById("book_title").value.trim();
    var summary = document.getElementById("book_summary").value.trim();
    var file = document.getElementById("file_buku").value.trim();
    
    if (!title || !summary || !file) return;
    
    document.getElementById("loading-indicator").style.display = "block";
}

// Cek apakah bagian hasil ada di halaman
if (document.getElementById("resultsList")) {

    // --- Elemen Global ---
    const searchInput = document.getElementById("searchInput");
    const sortSelect = document.getElementById("sortSelect");
    const resultsList = document.getElementById("resultsList");
    const btnAddIndex = document.getElementById("btnAddIndex");
    
    // --- Elemen Bulk Actions ---
    const btnBulkDelete = document.getElementById("btnBulkDelete");
    const btnBulkDownload = document.getElementById("btnBulkDownload");

    // --- Elemen Modal ---
    const modalOverlay = document.getElementById("indexModal");
    const modalTitle = document.getElementById("modal-title");
    const modalForm = document.getElementById("modal-form");
    const modalPhraseInput = document.getElementById("modal-phrase");
    const modalPagesInput = document.getElementById("modal-pages");
    const modalCloseBtn = document.getElementById("modalCloseBtn");
    const modalCancelBtn = document.getElementById("modal-cancel");
    const modalSearchBtn = document.getElementById("modal-search-phrase");
    const modalLoading = document.getElementById("modal-loading");

    // --- Cache & State ---
    let allItems = [];
    let nextId = 0;

    // --- Helper: Ambil item yang dicentang ---
    function getCheckedItems() {
        return Array.from(resultsList.querySelectorAll('.check-item:checked')).map(cb => cb.closest('li'));
    }

    // --- Fungsi Inisialisasi ---
    function initializeItems() {
        const initialItems = Array.from(resultsList.querySelectorAll("li:not(.no-results-server)"));
        allItems = initialItems.map(li => {
            const id = nextId++;
            li.dataset.id = id;
            
            const pagesSpan = li.querySelector('.pages');
            const fullPages = pagesSpan.getAttribute('title').replace('Halaman: ', '');
            li.dataset.fullPages = fullPages;
            
            return li;
        });
    }
    
    // --- Fungsi Filter & Render ---
    function filterAndSort() {
        if (!resultsList) return; 
        
        const query = searchInput.value.toLowerCase();
        const sortOrder = sortSelect.value;
        
        // 1. FILTER
        const filtered = allItems.filter(li =>
          li.querySelector(".phrase").textContent.toLowerCase().includes(query)
        );

        // 2. SORT
        filtered.sort((a, b) => {
          const textA = a.querySelector(".phrase").textContent.toLowerCase();
          const textB = b.querySelector(".phrase").textContent.toLowerCase();
          return sortOrder === "asc" ? textA.localeCompare(textB) : textB.localeCompare(textA);
        });

        // 3. RENDER
        resultsList.innerHTML = ""; 

        if (filtered.length === 0 && allItems.length > 0) {
          const notFoundLi = document.createElement("li");
          notFoundLi.textContent = "Indeks tidak ditemukan.";
          notFoundLi.className = "not-found"; 
          resultsList.appendChild(notFoundLi);
        } else if (allItems.length === 0) {
           const notFoundLi = document.createElement("li");
          notFoundLi.textContent = "Tidak ada frasa yang ditemukan.";
          notFoundLi.className = "no-results-server"; 
          resultsList.appendChild(notFoundLi);
        } else {
          filtered.forEach(li => resultsList.appendChild(li));
        }
    }

    // --- Fungsi Modal ---
    function openModal(mode, li = null) {
        modalOverlay.dataset.mode = mode;
        modalForm.reset(); 
        
        if (mode === 'edit') {
            modalTitle.textContent = "Edit Indeks";
            const phrase = li.querySelector(".phrase").textContent;
            const pages = li.dataset.fullPages; 
            
            modalPhraseInput.value = phrase;
            modalPagesInput.value = pages;
            modalOverlay.dataset.editingId = li.dataset.id;
            
        } else if (mode === 'add') {
            modalTitle.textContent = "Tambah Indeks";
            delete modalOverlay.dataset.editingId;
        }
        
        modalOverlay.style.display = 'flex'; 
        modalPhraseInput.focus(); 
    }
    
    function closeModal() {
        modalOverlay.style.display = 'none'; 
    }
    
    // --- Fungsi CRUD ---
    function createAndAddLi(phrase, pagesStr) {
        const newId = nextId++;
        const newLi = document.createElement("li");
        newLi.dataset.id = newId;
        newLi.dataset.fullPages = pagesStr;
        
        newLi.innerHTML = `
          <div class="index-content">
            <input type="checkbox" class="check-item" />
            <span class="phrase">${phrase}</span>
            <span class="pages" title="Halaman: ${pagesStr}">
              hlm. ${pagesStr}
            </span>
          </div>
          <div class="actions">
            <button class="btn-action btn-edit">✏️ Edit</button>
            <button class="btn-action btn-delete">🗑️ Hapus</button>
          </div>
        `;
        
        allItems.unshift(newLi);
    }
    
    function handleSave(e) {
        e.preventDefault(); 
        
        const mode = modalOverlay.dataset.mode;
        const phrase = modalPhraseInput.value.trim();
        const pagesStr = modalPagesInput.value.trim();
        
        if (!phrase) {
            alert("Frasa (Indeks) tidak boleh kosong.");
            return;
        }
        
        if (mode === 'add') {
            createAndAddLi(phrase, pagesStr || "?"); 
            
        } else if (mode === 'edit') {
            const editingId = modalOverlay.dataset.editingId;
            const liToUpdate = allItems.find(item => item.dataset.id == editingId);
            
            if (liToUpdate) {
                liToUpdate.querySelector(".phrase").textContent = phrase;
                liToUpdate.querySelector(".pages").textContent = `hlm. ${pagesStr}`;
                liToUpdate.querySelector(".pages").title = `Halaman: ${pagesStr}`;
                liToUpdate.dataset.fullPages = pagesStr;
            }
        }
        
        closeModal();
        filterAndSort(); 
    }

    // --- Bulk Delete ---
    if(btnBulkDelete) {
      btnBulkDelete.addEventListener('click', () => {
          const selectedLi = getCheckedItems();
          if (selectedLi.length === 0) return alert('Tidak ada indeks yang dipilih untuk dihapus.');
          
          if (!confirm(`Apakah Anda yakin ingin menghapus ${selectedLi.length} item ini?`)) return;

          const phrasesToDelete = selectedLi.map(li => li.querySelector('.phrase').textContent.trim());

          fetch('/api/bulk_delete', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ phrases: phrasesToDelete })
          })
          .then(response => response.json())
          .then(data => {
              if (data.status === 'success') {
                  const idsToDelete = new Set(selectedLi.map(li => li.dataset.id));
                  allItems = allItems.filter(item => !idsToDelete.has(item.dataset.id));
                  
                  alert(`Berhasil menghapus ${data.deleted_count} item.`);
                  filterAndSort();
              } else {
                  alert('Gagal menghapus: ' + data.message);
              }
          })
          .catch(err => {
              console.error(err);
              alert('Terjadi kesalahan koneksi.');
          });
      });
    }

    // --- Bulk Download PDF ---
    if(btnBulkDownload) {
      btnBulkDownload.addEventListener('click', () => {
          const selectedLi = getCheckedItems();
          if (selectedLi.length === 0) return alert('Tidak ada indeks yang dipilih untuk didownload.');

          const phrasesToDownload = selectedLi.map(li => li.querySelector('.phrase').textContent.trim());

          const originalText = btnBulkDownload.innerText;
          btnBulkDownload.innerText = "Memproses PDF...";
          btnBulkDownload.disabled = true;

          fetch('/api/download_selected_pdf', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ phrases: phrasesToDownload })
          })
          .then(response => {
              if (!response.ok) {
                  return response.json().then(err => { throw new Error(err.message || 'Gagal generate PDF') });
              }
              return response.blob();
          })
          .then(blob => {
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `indeks_terpilih_${new Date().getTime()}.pdf`;
              document.body.appendChild(a);
              a.click();
              a.remove();
              window.URL.revokeObjectURL(url);
          })
          .catch(err => {
              console.error(err);
              alert('Gagal mengunduh PDF: ' + err.message);
          })
          .finally(() => {
              btnBulkDownload.innerText = originalText;
              btnBulkDownload.disabled = false;
          });
      });
    }

    // --- Pencarian Halaman Otomatis (Modal) ---
    async function handleSearchPhrase() {
        const phrase = modalPhraseInput.value.trim();
        if (!phrase) {
            alert("Silakan masukkan frasa yang ingin dicari.");
            return;
        }
        
        modalLoading.style.display = 'inline';
        modalSearchBtn.disabled = true;
        
        try {
            const response = await fetch('/api/search_phrase', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phrase: phrase }),
            });
            
            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                modalPagesInput.value = result.pages.join(', ');
            } else {
                modalPagesInput.value = result.message || 'Error tidak diketahui';
            }
            
        } catch (error) {
            console.error("Error fetching search:", error);
            modalPagesInput.value = 'Gagal terhubung ke server.';
        } finally {
            modalLoading.style.display = 'none';
            modalSearchBtn.disabled = false;
        }
    }

    // --- Event Listeners ---
    resultsList.addEventListener("click", (e) => {
        const target = e.target;
        const li = target.closest("li");
        if (!li) return;
        
        if (target.classList.contains("btn-edit")) {
            openModal('edit', li);
        }
        
        if (target.classList.contains("btn-delete")) {
            if (confirm("Apakah Anda yakin ingin menghapus indeks ini?")) {
                const phrase = li.querySelector('.phrase').textContent.trim();
                
                fetch('/api/bulk_delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ phrases: [phrase] })
                })
                .then(res => res.json())
                .then(data => {
                    if(data.status === 'success') {
                        const idToDelete = li.dataset.id;
                        allItems = allItems.filter(item => item.dataset.id != idToDelete);
                        filterAndSort();
                    } else {
                        alert("Gagal menghapus: " + data.message);
                    }
                });
            }
        }
    });
    
    btnAddIndex.addEventListener("click", () => openModal('add'));
    searchInput.addEventListener("input", filterAndSort);
    sortSelect.addEventListener("change", filterAndSort);
    modalCloseBtn.addEventListener("click", closeModal);
    modalCancelBtn.addEventListener("click", closeModal);
    modalOverlay.addEventListener("click", (e) => {
        if (e.target === modalOverlay) closeModal();
    });
    modalForm.addEventListener("submit", handleSave);
    modalSearchBtn.addEventListener("click", handleSearchPhrase);

    // --- Inisialisasi ---
    initializeItems();
    filterAndSort(); 
}