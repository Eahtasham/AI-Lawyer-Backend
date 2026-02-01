from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import requests
import tarfile
import logging
import os
from typing import Generator

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

BASE_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com"

def get_index(year: str):
    """Download index file for the year"""
    url = f"{BASE_URL}/data/tar/year={year}/english/english.index.json"
    logger.info(f"Fetching index: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch index: {e}")
        return None

def find_tar_part(index_data, filename):
    """Find which TAR part contains the file"""
    if "parts" not in index_data:
        logger.error("Invalid index format: 'parts' not found")
        return None
        
    for part in index_data["parts"]:
        if filename in part["files"]:
            return part["name"]
            
    return None

def stream_file_from_tar(year: str, tar_name: str, target_filename: str) -> Generator[bytes, None, None]:
    """Stream TAR file and yield specific member content"""
    tar_url = f"{BASE_URL}/data/tar/year={year}/english/{tar_name}"
    logger.info(f"Streaming from: {tar_url}")
    
    try:
        # Stream the request
        with requests.get(tar_url, stream=True) as response:
            response.raise_for_status()
            
            # Open tar stream
            # mode='r|*' is essential for streaming (pipes)
            with tarfile.open(fileobj=response.raw, mode='r|*') as tar:
                for member in tar:
                    if member.name.endswith(target_filename):
                        logger.info(f"Found file: {member.name}")
                        
                        f = tar.extractfile(member)
                        if f:
                            while chunk := f.read(8192):
                                yield chunk
                            return
                
                logger.error(f"File {target_filename} not found in {tar_name}")
                raise FileNotFoundError(f"File {target_filename} not found in archive")

    except Exception as e:
        logger.error(f"Error during stream/extract: {e}")
        raise

@router.get("/download-judgement")
def download_judgement(url: str = Query(..., description="The URL or filename of the judgement PDF")):
    """
    Download a judgement PDF. Be robust to full URL or just filename input.
    """
    try:
        # Extract filename from URL if a full URL is provided
        filename = url.split("/")[-1]
        
        # Extract year from filename (assuming format like "1955_1_1_25_EN.pdf")
        parts = filename.split("_")
        if not parts or not parts[0].isdigit():
             raise HTTPException(status_code=400, detail="Invalid filename format. Could not extract year.")
        
        year = parts[0]
        
        # 1. Get Index
        index = get_index(year)
        if not index:
             raise HTTPException(status_code=404, detail=f"Index not found for year {year}")

        # 2. Find TAR part
        tar_part = find_tar_part(index, filename)
        if not tar_part:
             raise HTTPException(status_code=404, detail=f"File {filename} not found in index for year {year}")

        # 3. Stream Response
        return StreamingResponse(
            stream_file_from_tar(year, tar_part, filename),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Download endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
