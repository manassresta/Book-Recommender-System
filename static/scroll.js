var scrollTopBtn = document.getElementById("scrollTopBtn");

      // Show the button when the user scrolls down 100px from the top of the document
      window.onscroll = function () {
        if (
          document.body.scrollTop > 500 ||
          document.documentElement.scrollTop > 500
        ) {
          scrollTopBtn.style.display = "block";
        } else {
          scrollTopBtn.style.display = "none";
        }
      };

      // Smooth scroll to the top when the button is clicked
      scrollTopBtn.onclick = function () {
        window.scrollTo({
          top: 0,
          behavior: "smooth",
        });
      };