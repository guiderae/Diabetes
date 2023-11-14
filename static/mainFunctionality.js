

// This is an async call initiated by a click of the "Make Graph" button.  It makes an async call
// to the endpoint, /TrainAndTest.  Once the response from the endpoint is received,
// the returned HTML is placed into the UI by the displayGraphHTML() function
function makeGraph(){
    let workingMessage = "Working..........";
    let graphContainerObj = document.getElementById("graphContainer");
    graphContainerObj.innerHTML = "<img src='static/workingMsg.png'/>";
    let url = "/TrainAndTest";
    let formObj = document.getElementById("mainForm");
    let formData = new FormData(formObj);
    postGraphRequest(url, formData)
        .then(serializedImage => displayGraphHTML(serializedImage))
        .catch(error => console.error(error))
}
// Does the HTTPRequest to the endpoint given by the param, url.  The
// param formData is a dictionary that contains the values of all the input tags
// in the user interface.
async function postGraphRequest(url, formData){
    return fetch(url,{
        method: 'POST',
        body: formData
        })
        .then((response) => response.text());
}
// Takes the response from the param, serializedImage, builds an <img>
// tag that contains the serialized image, and assigns that <img>
// to the innerHTML attribute of the destination div.
function displayGraphHTML(serializedImage){
    let graphContainerObj = document.getElementById("graphContainer");
    graphContainerObj.style.display = "block";
    let graphHTML =  "<img src='data:image/png;base64, " + serializedImage +
        "' class='imgSize'/>" ;
    graphContainerObj.innerHTML = graphHTML;
}
// Add event listeners to radio buttons and to "Make Graph" button.

// Add EventListener to button.  The event is "click" and the listener is the
// function, makeGraph:
const makeGraphBtnObj = document.getElementById("makeGraphBtn");
makeGraphBtnObj.addEventListener("click", makeGraph);

