import React, { useState } from "react";
import Message from "./Message";
import axios from "axios";

const FileUpload = () => {
  const [file, setFile] = useState("");
  const [uploadedFile, setUploadedFile] = useState({});
  const [message, setMessage] = useState("");
  const [isLoading, setLoading] = useState(false);

  const validate = (selectedFile) => {
    if (!selectedFile) {
      return "*** Please Upload a File ***";
    }
    const one_megabyte = 1024 ** 2;
    const file_size_limit = 50;
    if (selectedFile.size > file_size_limit * one_megabyte) {
      return `*** File size exceeds ${file_size_limit}MB ***`;
    }
    let filename = selectedFile.name;
    let ext = filename.substring(filename.lastIndexOf(".") + 1);
    if (!(ext === "jpg" || ext === "png" || ext === "jpeg" || ext === "mp4")) {
      return "*** Unsupported Type (png or mp4) ***";
    }
    return "";
  };

  const onChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setMessage(validate(selectedFile));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (isLoading) return;

    setLoading(true);
    let errorMessage = validate(file);
    if (!errorMessage) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        console.log(formData.get("file"));
        const res = await axios.post("/upload", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        const { OriginFile,ProcessedFile } = res.data;
        setUploadedFile({ OriginFile,ProcessedFile });
        errorMessage = "*** File Successfully Uploaded ***";
      } catch (error) {
        errorMessage = "*** We will be back soon ***";
      }
    }
    setMessage(errorMessage);
    setLoading(false);
  };

  return (
    <div>
      {message && <Message msg={message} />}
      <form onSubmit={onSubmit}>
        <div className="custom-file mb-4 ">
          <input
            type="file"
            className="custom-file-input"
            id="customFile"
            onChange={onChange}
          />
        </div>

        <button
          className="btn btn-outline-primary btn-block btn-lg mt-4"
          disabled={isLoading}
          onSubmit={onSubmit}
        >
          {!isLoading ? (
            <span>Upload</span>
          ) : (
            <span>
              <i className="fas fa-spinner fa-spin"></i> Uploading...
            </span>
          )}
        </button>
      </form>
      {uploadedFile && (
        <div className="row mt-5">
          <div className="col-md-6 m-auto">
            <h3 className="text-center">{uploadedFile.fileName}</h3>
          </div>
        </div>
      )}

     
        {
        uploadedFile.ProcessedFile? <img src={`/static?filename=${uploadedFile.ProcessedFile}`}></img>:''
        }
    </div>
  );
};

export default FileUpload;


