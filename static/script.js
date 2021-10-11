// New skin on page load
getSkin()

 // Click buttons to generate new characters or download
document.getElementById("new").addEventListener("click", function() {
  getSkin();
  
})
document.getElementById("download").addEventListener("click", function() {
  download();
})


function getSkin() {
 async function postChar(url = '', data = {}) {
  const response = await fetch(url, {
    method: 'POST',
    mode: 'cors',
    cache: 'no-cache', 
    credentials: 'same-origin', 
    headers: {
      'Content-Type': 'application/json'
    },
    redirect: 'follow',
    referrerPolicy: 'no-referrer', 
    body: JSON.stringify(data)
  });
  let pngData = response;
  return pngData
}

postChar('/skin')
  .then(response => response.blob())
        .then(blob => {
    var url = window.URL.createObjectURL(blob)
    let Container = document.getElementById("imgbox");
    Container.src =  url;
    getRaw();
  });
}

function getRaw() {

 async function postPNG(url = '', data = {}) {
  const response = await fetch(url, {
    method: 'POST',
    mode: 'cors',
    cache: 'no-cache', 
    credentials: 'same-origin', 
    headers: {
      'Content-Type': 'application/json'
    },
    redirect: 'follow',
    referrerPolicy: 'no-referrer', 
    body: JSON.stringify(data)
  });
  let pngData = response;
  return pngData
}

postPNG('/download')
     .then(response => response.blob())
        .then(blob => {
            var url = window.URL.createObjectURL(blob)
    	    let Container = document.getElementById("raw");
    	    Container.src =  url; 
        });
}

function download(){

    var url = document.getElementById("raw").src;
    var a = document.getElementById("download-text");
    a.href = url;
    a.download = `test.png`; 
    a.click() 
}