const express = require('express');
const path = require('path');
const app = express();

// Serve all files in the 'public' folder
app.use(express.static(path.join(__dirname, 'public')));

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
