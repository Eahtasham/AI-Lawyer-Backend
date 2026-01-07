document.addEventListener('DOMContentLoaded', async () => {
    // Set Year
    document.getElementById('year').textContent = new Date().getFullYear();

    // Fetch Version
    try {
        const response = await fetch('/version');
        if (response.ok) {
            const data = await response.json();
            if (data.version) {
                document.getElementById('app-version').textContent = data.version;
            }
        }
    } catch (error) {
        console.error('Failed to fetch version:', error);
    }

    // Mobile Sidebar Interaction
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const sidebarOverlay = document.querySelector('.sidebar-overlay');
    const closeSidebar = document.querySelector('.close-sidebar');

    function toggleSidebar() {
        sidebar.classList.toggle('active');
        sidebarOverlay.classList.toggle('active');
        document.body.style.overflow = sidebar.classList.contains('active') ? 'hidden' : '';
    }

    if (menuToggle) {
        menuToggle.addEventListener('click', toggleSidebar);
    }

    if (closeSidebar) {
        closeSidebar.addEventListener('click', toggleSidebar);
    }

    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', toggleSidebar);
    }
});
