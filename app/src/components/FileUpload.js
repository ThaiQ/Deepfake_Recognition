import React, { Fragment, useState } from "react";
import Message from "./Message";
import axios from "axios";

const FileUpload = () => {
  const [file, setFile] = useState("");
  const [uploadedFile, setUploadedFile] = useState({});
  const [message, setMessage] = useState("");

  const onChange = (e) => {
    setFile(e.target.files[0]);
    let filename = e.target.files[0].name;
    let ext = filename.substring(filename.lastIndexOf(".") + 1);
    if (ext == "jpg" || ext == "png" || ext == "jpeg" || ext == "mp4")
      setMessage(null);
    else setMessage("jpg, png, jpeg, or mp4 only");
  };

  const onSubmit = async (e) => {
    try {
      e.preventDefault();
      let ext = file.name.substring(file.name.lastIndexOf(".") + 1);

      if (ext == "jpg" || ext == "png" || ext == "jpeg" || ext == "mp4") {
        const formData = new FormData();
        formData.append("file", file);
        const res = await axios.post("/upload", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        const { OriginFile,ProcessedFile } = res.data;
        setUploadedFile({ OriginFile,ProcessedFile });
        setMessage("File Successfully Uploaded");
      } else {
        setMessage("jpg, png, jpeg, or mp4 only");
      }
    } catch (err) {
      setMessage("Please upload file");
    }
  };

  return (
    <Fragment>
      {message ? <Message msg={message} /> : null}
      <form onSubmit={onSubmit}>
        <div className="custom-file mb-4">
          <input
            type="file"
            className="custom-file-input"
            id="customFile"
            onChange={onChange}
          />
          <label className="custom-file-label" htmlFor="customFile" />
        </div>

        <input
          type="submit"
          value="Upload"
          className="btn btn-primary btn-block mt-4"
        />
      </form>
      {uploadedFile ? (
        <div className="row mt-5">
          <div className="col-md-6 m-auto">
            <h3 className="text-center">{uploadedFile.fileName}</h3>
          </div>
        </div>
      ) : null}

        {
          uploadedFile.ProcessedFile? <img src={`/static?filename=${uploadedFile.ProcessedFile}`}></img>:''
        }

    </Fragment>
  );
};

export default FileUpload;
