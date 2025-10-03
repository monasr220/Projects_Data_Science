import cv2
import imutils
import numpy as np
import argparse
import os

class PersonDetector:
    def __init__(self):
        """Initialize the HOG descriptor for person detection."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect_people(self, frame):
        """
        Detect people in a frame and draw bounding boxes.
        
        Args:
            frame: Input image/frame
            
        Returns:
            frame: Frame with bounding boxes and labels
        """
        # Detect people in the frame
        bounding_boxes, weights = self.hog.detectMultiScale(
            frame, 
            winStride=(4, 4), 
            padding=(8, 8), 
            scale=1.03
        )
        
        # Draw bounding boxes and labels
        person_count = len(bounding_boxes)
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            # Draw rectangle around detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add person label
            cv2.putText(
                frame, 
                f'Person {i + 1}', 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                1
            )
        
        # Add status and count information
        cv2.putText(
            frame, 
            'Status: Detecting', 
            (10, 30), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.8, 
            (255, 0, 0), 
            2
        )
        cv2.putText(
            frame, 
            f'Total Persons: {person_count}', 
            (10, 60), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.8, 
            (255, 0, 0), 
            2
        )
        
        return frame
    
    def detect_from_camera(self, output_path=None):
        """Detect people from camera feed."""
        print('[INFO] Starting camera detection...')
        video = cv2.VideoCapture(0)
        
        if not video.isOpened():
            print('[ERROR] Could not open camera')
            return
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            ret, frame = video.read()
            if ret:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        print('Press "q" to quit...')
        while True:
            ret, frame = video.read()
            if not ret:
                print('[ERROR] Could not read frame from camera')
                break
            
            # Detect people in frame
            frame = self.detect_people(frame)
            
            # Write frame to output video if specified
            if writer is not None:
                writer.write(frame)
            
            # Display frame
            cv2.imshow('Person Detection', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print('[INFO] Camera detection completed')
    
    def detect_from_video(self, video_path, output_path=None):
        """Detect people from video file."""
        if not os.path.exists(video_path):
            print(f'[ERROR] Video file not found: {video_path}')
            return
        
        print(f'[INFO] Opening video: {video_path}')
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print('[ERROR] Could not open video file')
            return
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = video.get(cv2.CAP_PROP_FPS)
            ret, frame = video.read()
            if ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print('Processing video... Press "q" to quit early')
        frame_count = 0
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize frame for faster processing
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            
            # Detect people in frame
            frame = self.detect_people(frame)
            
            # Write frame to output video if specified
            if writer is not None:
                writer.write(frame)
            
            # Display frame
            cv2.imshow('Person Detection', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f'Processed {frame_count} frames...')
        
        # Clean up
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f'[INFO] Video processing completed. Processed {frame_count} frames')
    
    def detect_from_image(self, image_path, output_path=None):
        """Detect people from image file."""
        if not os.path.exists(image_path):
            print(f'[ERROR] Image file not found: {image_path}')
            return
        
        print(f'[INFO] Opening image: {image_path}')
        image = cv2.imread(image_path)
        
        if image is None:
            print('[ERROR] Could not load image')
            return
        
        # Resize image for better display
        image = imutils.resize(image, width=min(800, image.shape[1]))
        
        # Detect people in image
        result_image = self.detect_people(image)
        
        # Save result if output path is provided
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f'[INFO] Result saved to: {output_path}')
        
        # Display result
        cv2.imshow('Person Detection', result_image)
        print('Press any key to close...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Person Detection using HOG')
    parser.add_argument('-v', '--video', 
                       help='Path to video file')
    parser.add_argument('-i', '--image', 
                       help='Path to image file')
    parser.add_argument('-c', '--camera', 
                       action='store_true', 
                       help='Use camera for detection')
    parser.add_argument('-o', '--output', 
                       help='Path to output file (video/image)')
    
    return parser.parse_args()


def main():
    """Main function to run person detection."""
    args = parse_arguments()
    detector = PersonDetector()
    
    # Determine which detection method to use
    if args.camera:
        detector.detect_from_camera(args.output)
    elif args.video:
        detector.detect_from_video(args.video, args.output)
    elif args.image:
        detector.detect_from_image(args.image, args.output)
    else:
        print('[ERROR] Please specify input source: --camera, --video, or --image')
        print('Use --help for more information')


if __name__ == "__main__":
    main()