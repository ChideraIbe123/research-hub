export function initializeChatTriggers() {
  const chatTriggers = document.querySelectorAll('.chat-trigger');
  const sidebar = document.getElementById('chatSidebar');

  chatTriggers.forEach(trigger => {
    trigger.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      sidebar?.classList.remove('translate-x-full');
      sidebar?.classList.add('translate-x-0');
    });
  });

  document.addEventListener('chatClosed', () => {
    document.body.classList.remove('overflow-hidden');
  });
}