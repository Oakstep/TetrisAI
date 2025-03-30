const { spawn } = require('child_process');

// Function to run Python script
function runPython(playerName) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['xpcalc.py', playerName]);

        let result = '';
        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Error: ${data.toString()}`);
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                resolve(result);
            } else {
                reject(new Error(`Python process exited with code ${code}`));
            }
        });
    });
}

async function main(playerName) {
    try {
        const pythonOutput = await runPython(playerName);
        console.log('Python Output:', pythonOutput);

        // Parse the output if it's JSON
        const results = JSON.parse(pythonOutput);
        console.log('XP Calculation Results:', results);
    } catch (error) {
        console.error('Error running Python script:', error);
    }
}

// Replace "Enshi" with the player name you want to test
main("Enshi");