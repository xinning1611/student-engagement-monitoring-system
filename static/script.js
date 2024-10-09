document.getElementById("start-btn").onclick = function() {
    const statusDiv = document.getElementById("status");
    statusDiv.innerText = "Starting monitoring...";

    // Start monitoring
    fetch(`/start-monitoring`)
        .then(response => response.text())
        .then(data => {
            statusDiv.innerText = data;  // Update status with the response
        });
};

document.getElementById("stop-btn").onclick = function() {
    const statusDiv = document.getElementById("status");
    fetch('/stop-monitoring')
        .then(response => response.text())
        .then(data => {
            statusDiv.innerText = data;  // Update status with the response
        });
};
