#include "ofApp.h"

/*
    If you are struggling to get the device to connect ( especially Windows Users )
    please look at the ReadMe: in addons/ofxKinect/README.md
*/

using namespace cv;
using namespace ofxCv;


//--------------------------------------------------------------
void ofApp::setup() {
	ofSetLogLevel(OF_LOG_VERBOSE);
    ofSetFrameRate(60);
    
	// enable depth->video image calibration
	kinect.setRegistration(true);
    
	kinect.init();
	//kinect.init(true); // shows infrared instead of RGB video image
	//kinect.init(false, false); // disable video image (faster fps)
	
	kinect.open();		// opens first available kinect
	//kinect.open(1);	// open a kinect by id, starting with 0 (sorted by serial # lexicographically))
	//kinect.open("A00362A08602047A");	// open a kinect using it's unique serial #
	
	// print the intrinsic IR sensor values
	if(kinect.isConnected()) {
		ofLogNotice() << "sensor-emitter dist: " << kinect.getSensorEmitterDistance() << "cm";
		ofLogNotice() << "sensor-camera dist:  " << kinect.getSensorCameraDistance() << "cm";
		ofLogNotice() << "zero plane pixel size: " << kinect.getZeroPlanePixelSize() << "mm";
		ofLogNotice() << "zero plane dist: " << kinect.getZeroPlaneDistance() << "mm";
	}
	
#ifdef USE_TWO_KINECTS
	kinect2.init();
	kinect2.open();
#endif
	
    grayMat = Mat( kinect.height,kinect.width,CV_8UC1);
    grayThresh = Mat(kinect.height,kinect.width,CV_8UC1);

    nearThreshold_mm = 500; //230;
    farThreshold_mm = 1500; //70;
	bThreshWithOpenCV = true;
	

    roi_rect = ofRectangle(10,10,kinect.width-20,kinect.height-20);
	// zero the tilt on startup
	angle = 0;
	kinect.setCameraTiltAngle(angle);
    kinect.setDepthClipping(500,4000);
    
	// start from the front
	bDrawPointCloud = false;
    
    display_scaler = 0.7;
    bSaveCloud = false;
}

//--------------------------------------------------------------
void ofApp::update() {
	
	ofBackground(100, 100, 100);
	
	kinect.update();
	
	// there is a new frame and we are connected
	if(kinect.isFrameNew()) {
		
		// load grayscale depth image from the kinect source
//		grayImage.setFromPixels(kinect.getDepthPixels());
        grayMat = toCv(kinect.getDepthPixels()); 
        
        //mask out the edges of the image to not track on walls
        cv::Point p1;
        cv::Point p2;

        //left
        p1 = cv::Point(0,0);
        p2 = cv::Point(roi_rect.getLeft(),kinect.getHeight());
        cv::rectangle(grayMat, p1, p2 , cvScalar(0, 0, 0), -1, 8, 0);
        //right
        p1 = cv::Point(roi_rect.getRight(),0);
        p2 = cv::Point(kinect.getWidth(),kinect.getHeight());
        cv::rectangle(grayMat, p1, p2 , cvScalar(0, 0, 0), -1, 8, 0);
        //top
        p1 = cv::Point(roi_rect.getLeft(),0);
        p2 = cv::Point(roi_rect.getRight(),roi_rect.getTop());
        cv::rectangle(grayMat, p1, p2 , cvScalar(0, 0, 0), -1, 8, 0);
        //bottom
        p1 = cv::Point(roi_rect.getLeft(),roi_rect.getBottom());
        p2 = cv::Point(roi_rect.getRight(),kinect.getWidth());
        cv::rectangle(grayMat, p1, p2 , cvScalar(0, 0, 0), -1, 8, 0);
        
        nearThreshold = ofMap(nearThreshold_mm,kinect.getFarClipping(), kinect.getNearClipping(),0,255);
        farThreshold = ofMap(farThreshold_mm,kinect.getFarClipping(), kinect.getNearClipping(),0,255);
        
        inRange(grayMat, Scalar(farThreshold), Scalar(nearThreshold), grayThresh);

		
		// find contours which are between the size of 20 pixels and 1/3 the w*h pixels.
		// also, find holes is set to true so we will get interior contours as well....
        contourFinder.setThreshold(0);
        contourFinder.findContours(grayThresh); //, 10, (kinect.width*kinect.height)/2, 20, false);
	}
	
#ifdef USE_TWO_KINECTS
	kinect2.update();
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	
	ofSetColor(255, 255, 255);
	
	if(bDrawPointCloud) {
		easyCam.begin();
		drawPointCloud();
		easyCam.end();
	} else {
        ofPushMatrix();
        ofScale(display_scaler);
		// draw from the live kinect
        kinect.drawDepth(0, 0); //, 400, 300);
        ofNoFill();
        ofDrawRectangle(roi_rect);
        
        kinect.draw(640, 0);
		
       
        drawMat(grayThresh,0, 480); 
        ofPushMatrix();
        ofTranslate(0, 480);
        contourFinder.draw(); 
        ofPopMatrix();
        
        ofPopMatrix();
        
#ifdef USE_TWO_KINECTS
        kinect2.draw(640, 480); 
#endif
	}
	
	// draw instructions
	ofSetColor(255, 255, 255);
	stringstream reportStream;
        
    if(kinect.hasAccelControl()) {
        reportStream << "accel is: " << ofToString(kinect.getMksAccel().x, 2) << " / "
        << ofToString(kinect.getMksAccel().y, 2) << " / "
        << ofToString(kinect.getMksAccel().z, 2) << endl;
    } else {
        reportStream << "Note: this is a newer Xbox Kinect or Kinect For Windows device," << endl
		<< "motor / led / accel controls are not currently supported" << endl << endl;
    }
    
	reportStream << "press p to switch between images and point cloud, rotate the point cloud with the mouse" << endl
	<< "using opencv threshold = " << bThreshWithOpenCV <<" (press spacebar)" << endl
	<< "set near threshold in mm " << nearThreshold_mm<< " (press: + -)" << endl
	<< "set far threshold in mm " << farThreshold_mm << " (press: < >) num blobs found " << contourFinder.size()
	<< ", fps: " << ofGetFrameRate() << endl
	<< "press c to close the connection and o to open it again, connection is: " << kinect.isConnected() << endl;

    if(kinect.hasCamTiltControl()) {
    	reportStream << "press UP and DOWN to change the tilt angle: " << angle << " degrees" << endl
        << "press 1-5 & 0 to change the led mode" << endl;
    }
    reportStream << "drag a rectangle over depth video to select ROI: " <<endl;
    
	ofDrawBitmapString(reportStream.str(), 20, 750);
    
}

void ofApp::drawPointCloud() {
	int w = 640;
	int h = 480;
	ofMesh mesh;
//	mesh.setMode(OF_PRIMITIVE_POINTS);
	int step = 2;
	for(int y = 0; y < h; y += step) {
		for(int x = 0; x < w; x += step) {
        
			if(roi_rect.inside(x, y) && kinect.getDistanceAt(x, y) > nearThreshold_mm && kinect.getDistanceAt(x, y) < farThreshold_mm) {
				mesh.addColor(kinect.getColorAt(x,y));
				mesh.addVertex(kinect.getWorldCoordinateAt(x, y));
			}
		}
	}
	glPointSize(3);
	ofPushMatrix();
	// the projected points are 'upside down' and 'backwards' 
	ofScale(1, -1, -1);
	ofTranslate(0, 0, -1000); // center the points a bit
	ofEnableDepthTest();
	mesh.drawVertices();
    
    if(bSaveCloud == true){
        bSaveCloud = false;
        mesh.save("pointCloud.ply");
    }
	ofDisableDepthTest();
	ofPopMatrix();
}

//--------------------------------------------------------------
void ofApp::exit() {
	kinect.setCameraTiltAngle(0); // zero the tilt on exit
	kinect.close();
	
#ifdef USE_TWO_KINECTS
	kinect2.close();
#endif
}

//--------------------------------------------------------------
void ofApp::keyPressed (int key) {
    if(key == 's'){
        bSaveCloud = true;
    }

	switch (key) {
		case ' ':
			bThreshWithOpenCV = !bThreshWithOpenCV;
			break;
			
		case'p':
			bDrawPointCloud = !bDrawPointCloud;
			break;
			
		case '>':
		case '.':
            farThreshold_mm +=5;
			if (farThreshold_mm > 4000) farThreshold_mm = 4000;
			break;
			
		case '<':
		case ',':
            farThreshold_mm -=5;
			if (farThreshold_mm < 500) farThreshold_mm = 500;
			break;
			
		case '+':
		case '=':
            nearThreshold_mm +=5;
			if (nearThreshold_mm > 4000) nearThreshold_mm = 4000;
			break;
			
		case '-':
            nearThreshold_mm -=5;
			if (nearThreshold_mm < 500) nearThreshold_mm = 500;
			break;
			
		case 'w':
			kinect.enableDepthNearValueWhite(!kinect.isDepthNearValueWhite());
			break;
			
		case 'o':
			kinect.setCameraTiltAngle(angle); // go back to prev tilt
			kinect.open();
			break;
			
		case 'c':
			kinect.setCameraTiltAngle(0); // zero the tilt
			kinect.close();
			break;
			
		case '1':
			kinect.setLed(ofxKinect::LED_GREEN);
			break;
			
		case '2':
			kinect.setLed(ofxKinect::LED_YELLOW);
			break;
			
		case '3':
			kinect.setLed(ofxKinect::LED_RED);
			break;
			
		case '4':
			kinect.setLed(ofxKinect::LED_BLINK_GREEN);
			break;
			
		case '5':
			kinect.setLed(ofxKinect::LED_BLINK_YELLOW_RED);
			break;
			
		case '0':
			kinect.setLed(ofxKinect::LED_OFF);
			break;
			
		case OF_KEY_UP:
			angle++;
			if(angle>30) angle=30;
			kinect.setCameraTiltAngle(angle);
			break;
			
		case OF_KEY_DOWN:
			angle--;
			if(angle<-30) angle=-30;
			kinect.setCameraTiltAngle(angle);
			break;
	}
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button)
{
    if(bDrawPointCloud == false){
    roi_rect.setWidth((x/display_scaler) - roi_rect.getX());
    roi_rect.setHeight((y/display_scaler) - roi_rect.getY());
    }
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button)
{
    if(bDrawPointCloud == false){
    roi_rect.setX(x/display_scaler);
    roi_rect.setY(y/display_scaler);
    }
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button)
{

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h)
{

}
