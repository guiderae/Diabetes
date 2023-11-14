    /*
        Reads the prediction data csv and puts the data into a table that is displayed in the UI.
        The table is dynamically built in the DOM.
        The file name of the data file is stored in a hidden element named 'selectedFileName' for later use.
     */
  function handleFileSelect(evt){
        let files = evt.target.files; // FileList object
        let fileObj = files[0];  // Single file selection
        let selectedFileName = files[0].name;
        let hiddenFileNameObj = document.getElementById("selectedFileName");
        hiddenFileNameObj.value = selectedFileName;
        let reader = new FileReader();

        reader.onload = function(e) {
            let predDataTableObj = document.getElementById("predDataTable");
            let textArr = this.result.split('\n');
            if(textArr.length > 1){  // Ensure at least one row of data with column headers
                let headingRow = textArr[0];
                let headings = headingRow.split(',');
                let trElement = document.createElement('tr');
                for (let headingIndex = 0; headingIndex < headings.length; headingIndex++){
                    let thElement = document.createElement('th');
                    thElement.innerHTML = headings[headingIndex];
                    trElement.appendChild(thElement);
                }
                predDataTableObj.appendChild(trElement);
                for(let i = 1; i < textArr.length; i++){
                    let dataRow = textArr[i].split(',');
                    let trDataElement = document.createElement('tr');
                    for(let j = 0; j < dataRow.length; j++){
                        let tdDataElement = document.createElement('td');
                        tdDataElement.innerHTML = dataRow[j];
                        trDataElement.appendChild(tdDataElement);
                    }
                    predDataTableObj.appendChild(trDataElement);
                }
            }
        }
        reader.readAsText(fileObj);
        document.getElementById('predictBtn').style.display = 'block';  // display Prediction Btn
  }
  /*
    Handles click of the prediction button.  Creates an async post to the URL "/calculatePredictions".
    Receives a json response from the server and adds the prediction data to the corresponding data
    row already displayed in a table in the UI.
   */
  function handlePredictSelect(){
    let url = "/calculatePredictions";
    let formObj = document.getElementById("mainForm");
    let formData = new FormData(formObj);  // Only one input element in form, the hidden field

    postPredictionRequest(url, formData)
        .then(serializedResults => insertResults(serializedResults))
        .catch(error => console.error(error))
  }
  async function postPredictionRequest(url, formData){
    return fetch(url,{
        method: 'POST',
        body: formData
        })
        .then((response) => response.json());
  }
  /*
    Iterates through the three new columns of data and calls addColumn() to append the new columns
    into the already displayed data table.
    Param jsonData is a list of dictionaries, where each dictionary contains a column of data including
    the table heading.
   */
  function insertResults(jsonData){ // jsonData is a list of dict
    let keys = Object.keys(jsonData[0]);  // All dicts have the same dict keys.   All dicts have 3 keys:
                                          // "pred", "nodprob", "yesdprob"
    for(let i in keys){
        addColumn(i, jsonData, keys[i]);
    }

  }
  /*
    Adds a column of data with header to table
    Param colNo is the number of the column in the json data.
    Param jsonData is an array of dictionaries.  Each dict has 3 entries, one for each new column.
            Note: There are no column labels in this data.
    Param jsonKey is the dict key for the rows of data.  All rows have the same 3 keys.
   */
  const addColumn = (colNo, jsonData, jsonKey) => {
    let headingList = ["Prediction", "Probability No", "Probability Yes"]; // headers for new columns
    let trList = document.querySelectorAll('#predDataTable tr'); // Get all <tr> in table
    trList.forEach((row, i) => {  // For each table row, add column data
        const cell = document.createElement(i ? "td" : "th")  // Anything other than 0 is a tr
        if(i == 0){  // th header row
            cell.innerHTML = headingList[colNo]
        }else{  // td data row
            cell.innerHTML = jsonData[i-1][jsonKey]; // The var i is a zero based index for the existing table
                                                     // that contains column headers.  So we must use i-1 as the
                                                    // json data index since jsonData has no column headers.
            if(jsonKey == "pred"){  // Color "Prediction" value red if positively predicted
                if(cell.innerHTML == "Diabetes Predicted") {
                    cell.className = "redColor"
                }
            }
        }
       row.appendChild(cell)
    });
 }
  document.getElementById('loadDataBtn').addEventListener('change', handleFileSelect, false);
  document.getElementById('predictBtn').addEventListener('click', handlePredictSelect, false);