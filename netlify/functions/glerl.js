const https = require('https');

exports.handler = async function(event) {
  const today = new Date();
  const y  = today.getUTCFullYear();
  const mm = String(today.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(today.getUTCDate()).padStart(2, '0');
  const url = `https://www.glerl.noaa.gov/metdata/chi/${y}/${y}${mm}${dd}.04t.txt`;

  return new Promise((resolve) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        resolve({
          statusCode: 200,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'text/plain',
            'Cache-Control': 'no-cache'
          },
          body: data
        });
      });
    }).on('error', (err) => {
      resolve({
        statusCode: 500,
        headers: { 'Access-Control-Allow-Origin': '*' },
        body: `Error: ${err.message}`
      });
    });
  });
};
```