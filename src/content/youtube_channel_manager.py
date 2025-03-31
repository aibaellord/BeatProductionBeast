#!/usr/bin/env python3
"""
YouTubeChannelManager: A comprehensive module for automating YouTube channel operations
with sacred geometry principles for optimal content scheduling and engagement.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from ..utils.sacred_geometry_core import SacredGeometryCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeChannelManager:
    """
    A comprehensive YouTube channel management class that handles authentication, 
    video uploads, content scheduling based on sacred geometry timing,
    analytics tracking, and comment management with automated responses.
    """
    
    # YouTube API scopes required for full channel management
    SCOPES = [
        'https://www.googleapis.com/auth/youtube',
        'https://www.googleapis.com/auth/youtube.force-ssl',
        'https://www.googleapis.com/auth/youtube.upload',
        'https://www.googleapis.com/auth/youtube.readonly',
        'https://www.googleapis.com/auth/youtubepartner'
    ]
    
    # YouTube API service name and version
    API_SERVICE_NAME = 'youtube'
    API_VERSION = 'v3'
    
    def __init__(
        self, 
        credentials_file: str, 
        token_file: str, 
        channel_id: Optional[str] = None,
        use_sacred_timing: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize the YouTube Channel Manager.
        
        Args:
            credentials_file: Path to the client_secret.json file from Google API Console
            token_file: Path to store the OAuth 2.0 token
            channel_id: YouTube channel ID (optional if using authenticated user's channel)
            use_sacred_timing: Whether to use sacred geometry for content scheduling
            log_level: Logging level
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.channel_id = channel_id
        self.use_sacred_timing = use_sacred_timing
        self.youtube = None
        self.sacred_geometry = SacredGeometryCore()
        
        # Set logging level
        logger.setLevel(log_level)
        
        # Authenticate with YouTube API
        self._authenticate()
        
        if self.youtube and not channel_id:
            try:
                # Get authenticated user's channel ID if not provided
                channels_response = self.youtube.channels().list(
                    part='id',
                    mine=True
                ).execute()
                
                if channels_response['items']:
                    self.channel_id = channels_response['items'][0]['id']
                    logger.info(f"Using authenticated user's channel ID: {self.channel_id}")
                else:
                    logger.warning("No channels found for authenticated user")
            except googleapiclient.errors.HttpError as e:
                logger.error(f"Failed to get user channel: {e}")
    
    def _authenticate(self) -> None:
        """
        Authenticate with YouTube API using OAuth 2.0.
        
        Returns:
            None
        """
        creds = None
        
        # Check if token file exists
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as token:
                    creds = Credentials.from_authorized_user_info(
                        json.load(token), 
                        self.SCOPES
                    )
            except Exception as e:
                logger.error(f"Error loading credentials from token file: {e}")
        
        # If credentials don't exist or are invalid, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Error refreshing credentials: {e}")
                    creds = None
            
            # If still no valid credentials, run the OAuth flow
            if not creds:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, 
                        self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    
                    # Save the credentials for the next run
                    with open(self.token_file, 'w') as token:
                        token.write(creds.to_json())
                    logger.info("New credentials saved to token file")
                except Exception as e:
                    logger.error(f"Error during OAuth flow: {e}")
                    raise
        
        try:
            # Build the YouTube API service
            self.youtube = googleapiclient.discovery.build(
                self.API_SERVICE_NAME, 
                self.API_VERSION, 
                credentials=creds
            )
            logger.info("Successfully authenticated with YouTube API")
        except Exception as e:
            logger.error(f"Failed to build YouTube API service: {e}")
            raise
    
    def upload_video(
        self, 
        video_file: str, 
        title: str, 
        description: str, 
        tags: List[str], 
        category_id: str = "22",  # 22 is 'People & Blogs'
        privacy_status: str = "public",
        publish_at: Optional[datetime] = None,
        thumbnail_file: Optional[str] = None,
        language: str = "en",
        notify_subscribers: bool = True,
        made_for_kids: bool = False
    ) -> Optional[str]:
        """
        Upload a video to YouTube with optimized metadata.
        
        Args:
            video_file: Path to the video file
            title: Video title
            description: Video description
            tags: List of tags for the video
            category_id: YouTube category ID
            privacy_status: Privacy status ('public', 'private', 'unlisted')
            publish_at: Schedule publishing time (for private videos)
            thumbnail_file: Path to thumbnail image file
            language: Content language
            notify_subscribers: Whether to notify subscribers
            made_for_kids: Whether content is made for kids
            
        Returns:
            Video ID if upload successful, None otherwise
        """
        if not self.youtube:
            logger.error("YouTube API client not initialized")
            return None
        
        if not os.path.exists(video_file):
            logger.error(f"Video file not found: {video_file}")
            return None
        
        # Prepare video metadata
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': category_id,
                'defaultLanguage': language
            },
            'status': {
                'privacyStatus': privacy_status,
                'selfDeclaredMadeForKids': made_for_kids,
                'notifySubscribers': notify_subscribers
            }
        }
        
        # Add scheduled publishing if provided
        if publish_at and privacy_status == 'private':
            # Format datetime according to ISO 8601
            body['status']['publishAt'] = publish_at.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        try:
            # Build request to upload the video
            request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=googleapiclient.http.MediaFileUpload(
                    video_file, 
                    resumable=True,
                    chunksize=1024*1024  # Upload in 1MB chunks
                )
            )
            
            # Execute upload and track progress
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Uploaded {int(status.progress() * 100)}%")
            
            video_id = response['id']
            logger.info(f"Video uploaded successfully. Video ID: {video_id}")
            
            # Set thumbnail if provided
            if thumbnail_file and os.path.exists(thumbnail_file):
                self.set_thumbnail(video_id, thumbnail_file)
            
            return video_id
            
        except googleapiclient.errors.HttpError as e:
            logger.error(f"An HTTP error occurred during upload: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during upload: {e}")
            return None
    
    def set_thumbnail(self, video_id: str, thumbnail_file: str) -> bool:
        """
        Set a custom thumbnail for a video.
        
        Args:
            video_id: YouTube video ID
            thumbnail_file: Path to thumbnail image file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(thumbnail_file):
            logger.error(f"Thumbnail file not found: {thumbnail_file}")
            return False
        
        try:
            self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=googleapiclient.http.MediaFileUpload(
                    thumbnail_file, 
                    resumable=False
                )
            ).execute()
            logger.info(f"Thumbnail set for video {video_id}")
            return True
        except googleapiclient.errors.HttpError as e:
            logger.error(f"An HTTP error occurred setting thumbnail: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred setting thumbnail: {e}")
            return False
    
    def get_optimal_upload_time(
        self, 
        base_date: datetime = None, 
        days_ahead: int = 7
    ) -> datetime:
        """
        Calculate the optimal upload time based on sacred geometry principles.
        
        Args:
            base_date: Starting date for calculation (defaults to now)
            days_ahead: Number of days to look ahead
            
        Returns:
            Optimal datetime for uploading
        """
        if not self.use_sacred_timing:
            # If not using sacred timing, return a reasonable time (e.g., 3pm tomorrow)
            if base_date is None:
                base_date = datetime.now()
            return base_date + timedelta(days=1, hours=15 - base_date.hour)
        
        # Use sacred geometry to determine optimal time
        if base_date is None:
            base_date = datetime.now()
        
        # Calculate golden ratio time points over the next few days
        optimal_times = []
        for day in range(days_ahead):
            day_start = base_date + timedelta(days=day)
            day_start = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get golden ratio points throughout the day (phi-based time points)
            phi_times = self.sacred_geometry.generate_phi_time_points(
                start_time=day_start,
                num_points=8,  # Generate 8 potential times per day
                day_fraction=True  # Use day-based calculation
            )
            
            optimal_times.extend(phi_times)
        
        # Calculate Schumann resonance alignment
        best_time = None
        best_score = -1
        
        for time_point in optimal_times:
            # Skip times in the past
            if time_point < datetime.now():
                continue
                
            # Calculate alignment score based on multiple factors
            score = self.sacred_geometry.calculate_time_resonance(time_point)
            
            if score > best_score:
                best_score = score
                best_time = time_point
        
        if best_time is None:
            # Fallback to a reasonable time if no optimal time found
            best_time = base_date + timedelta(days=1, hours=15 - base_date.hour)
            
        logger.info(f"Optimal upload time calculated: {best_time} (score: {best_score})")
        return best_time
    
    def schedule_content(
        self, 
        video_file: str,
        title: str,
        description: str,
        tags: List[str],
        target_date: Optional[datetime] = None,
        thumbnail_file: Optional[str] = None,
        category_id: str = "22"
    ) -> Optional[str]:
        """
        Schedule content for upload at an optimal time based on sacred geometry.
        
        Args:
            video_file: Path to the video file
            title: Video title
            description: Video description
            tags: List of tags for the video
            target_date: Target date for publishing (will be optimized)
            thumbnail_file: Path to thumbnail image file
            category_id: YouTube category ID
            
        Returns:
            Video ID if scheduling successful, None otherwise
        """
        # Calculate optimal upload time
        if target_date is None:
            target_date = datetime.now()
        
        optimal_time = self.get_optimal_upload_time(base_date=target_date)
        
        # Upload as private first, then schedule for the optimal time
        try:
            video_id = self.upload_video(
                video_file=video_file,
                title=title,
                description=description,
                tags=tags,
                category_id=category_id,
                privacy_status="private",  # Start as private
                publish_at=optimal_time,   # Schedule for optimal time
                thumbnail_file=thumbnail_file,
                notify_subscribers=True
            )
            
            if video_id:
                logger.info(f"Video {video_id} scheduled for {optimal_time}")
                return video_id
            else:
                logger.error("Failed to schedule video")
                return None
                
        except Exception as e:
            logger.error(f"An error occurred during content scheduling: {e}")
            return None
    
    def get_channel_analytics(
        self, 
        metrics: List[str] = None, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get analytics data for the channel.
        
        Args:
            metrics: List of metrics to retrieve (views, likes, comments, etc.)
            start_date: Start date for analytics
            end_date: End date for analytics
            
        Returns:
            Dictionary containing analytics data
        """
        if not self.youtube:
            logger.error("YouTube API client not initialized")
            return {}
            
        if not self.channel_id:
            logger.error("Channel ID not set")
            return {}
        
        if metrics is None:
            metrics = ['views', 'likes', 'subscribersGained', 'comments', 'shares', 'watchTime']
            
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=28)
            
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        

