clear; clc; close all;

% Enable camera, onboard surface pro rear camera.
cam = webcam("Microsoft Camera Rear");
% Keep video feed
preview(cam);

% Start image acquisition, processing, and output
while (1)
    % Clear previo us output
    close;
    % Wait for camera to adjust exposure
    pause(2);
    % Take snapshot
    img = snapshot(cam);
    % Check if the there is a card in camera
    [counts, binLocations] = imhist(img);
    histCount = 0;
    for i = 1:length(imhist(img))
        if counts(i) > 50
            histCount = histCount + 1;
        end
    end
    % check if img has a card in it using histrogram
    if histCount > 20
        I = img;
    else
        I = I;
    end

    % Binarize image, black and white, only with 0s and 1s
    I_b = I;
    if size(I, 3) > 1
        I_b = rgb2gray(I);
    end
    I_b = imbinarize(I_b);
    % Display intermediate steps
    figure(1)
    imhist(I);
    title('Input image histogram')
    pause(0.2)
    % Resize all images to the same size
    I_b = imresize((I_b),900 / size(I_b, 1),'nearest');
    
    % Transform the cards for collecting ranks and suits training dataset
    % transform = randomAffine2d('Scale', [0.9, 1.1], 'Rotation', [-30 30]);
    % I_bs = imwarp(I_b,transform);
    
    % Rotate image from 0 to 180 degrees counter clock wise
    % At 1 degree interval
    degree = 180;
    total = zeros(1,degree);
    for c = 1:1:degree
        countWhite = 0;
        Ib_r = imrotate(I_b,c);
        
        % Sum white column/stem values
        % Look for the lowest column count
        % Threshold column sum set to 20 pixels
        s = sum(Ib_r);
        for i = 1:1:length(s)
            if s(i) > 20
                countWhite = countWhite + 1;
            end
        end
        total(c) = countWhite;
    end
    % Display intermediate steps
    figure(1)
    histogram(total);
    pause(0.2)
    % Find the index/degree which gives the least count of vertical whites
    % Rotate the image by that index/degree: base crop image
    [least, indx] = min(total);
    I_ur_bi = imrotate(I_b, indx);
    
    % Find stem pulse values of Ibase, summing all the 1s/whites
    % Finding the left and right coordinates
    % Find which column has a sudden jump in white pixels greater than 200
    Isum_upright = sum(I_ur_bi);
    urleft_tmp = length(Isum_upright(1:find(Isum_upright > 200,1)));
    
    Ileft  = urleft_tmp + 4;
    Iright = Ileft + least;
    
    % Same method to find top and bottom crop points
    I_HoriTop = imrotate(I_ur_bi, 90);
    Isum_Horizontal_Top = sum(I_HoriTop);
    horiTop_tmp = length(Isum_Horizontal_Top(1:find(Isum_Horizontal_Top > 200,1)));
    
    I_HoriBottom = imrotate(I_ur_bi, -90);
    Isum_Horizontal_Bottom = sum(I_HoriBottom);
    horiBottom_tmp = length(Isum_Horizontal_Bottom(1:find(Isum_Horizontal_Bottom > 200,1)));
    
    Itop    = horiTop_tmp;
    Ibottom = length(Isum_Horizontal_Bottom) - horiBottom_tmp;
    
    Icrop = I_ur_bi(Itop:Ibottom,Ileft:Iright,:);
    % Display intermediate steps
    figure(1)
    imshow(Icrop);
    pause(0.2)
    
    % Higher resolution rank and suit crops for display
    [Iheight, Iwidth] = size(Icrop);
    
    % Using actual card print ratio
    Iheight_r = 2.1/8.7;
    Iwidth_r = 0.85/6.2;
    
    Is_right = round(Iwidth * Iwidth_r)+3;
    Is_bottom = round(Iheight * Iheight_r);
    
    Itargetcrop = Icrop(11:Is_bottom,11:(Is_right-5),:);
    Itargetheight = length(imrotate(Itargetcrop,90));
    
    I_rankbottom = round(Itargetheight * 115674 / 200000);
    I_rankcrop = Itargetcrop(1:I_rankbottom,:,:);
    
    I_suittop = Itargetheight - I_rankbottom;
    I_suitcrop = Itargetcrop(I_rankbottom:Itargetheight,:,:);
    %
    % Lower resolution crops of ranks and suits to train and ID by CNN
    Icrop_n = imresize((Icrop), 200 / size(Icrop, 1));
    Icrop_n = 1. - Icrop_n;
    image_rank = imcrop(Icrop_n, [4 3 18 29]);
    image_suit = imcrop(Icrop_n, [5 30 14 21]);
    image_rank = image_rank*255;
    image_suit = image_suit*255;
    
    % Collecting training datasets for CNN
    %-------------------------------------
    %{
    n = 1:30;
    name = "rank"+int2str(k*100+n)+".bmp";
    switch img(1)
        case '1'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\10'
        case '2'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\2'
        case '3'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\3'
        case '4'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\4'
        case '5'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\5'
        case '6'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\6'
        case '7'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\7'
        case '8'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\8'
        case '9'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\9'
        case 'J'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\J'
        case 'Q'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\Q'
        case 'K'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\K'
        case 'A'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_ranks\rank_dataset\A'
    end
    
    fullDestinationFileName = fullfile(destinationFolder, name);
    imwrite(image_rank, fullDestinationFileName);
    
    name = "suit"+int2str(k*100+n)+".bmp";
    
    switch img(2)
        case 'S'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_suits\suit_dataset\Spades'
        case 'C'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_suits\suit_dataset\Clubs'
        case 'H'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_suits\suit_dataset\Hearts'
        case 'D'
            destinationFolder = 'C:\Alex\university\Fall 2021\image processing\proj3\machine_learning_suits\suit_dataset\Diamonds'
    end
    fullDestinationFileName2 = fullfile(destinationFolder, name);
    imwrite(image_suit, fullDestinationFileName2);
    %}
    % --------------------------------
    
    % Using CNN to determine ranks and suits
    load suit_net
    label_suit = classify(suit_net, image_suit);
    load rank_net
    label_rank = classify(rank_net, image_rank);
    
    % For display purpose of named ranks and suits
    suit = convertCharsToStrings(char(label_suit(1)));
    rank = '';
    rank_t = convertCharsToStrings(char(label_rank(1)));
    
    if rank_t == 'A'
        rank = 'Ace';
    elseif rank_t == "2"
        rank = "Two";
    elseif rank_t == "3"
        rank = "Three";
    elseif rank_t == "4"
        rank = "Four";
    elseif rank_t == "5"
        rank = "Five";
    elseif rank_t == "6"
        rank = "Six";
    elseif rank_t == "7"
        rank = "Seven";
    elseif rank_t == "8"
        rank = "Eight";
    elseif rank_t == "9"
        rank = "Nine";
    elseif rank_t == "10"
        rank = "Ten";
    elseif rank_t == "J"
        rank = "Jack";
    elseif rank_t == "Q"
        rank = "Queen";
    elseif rank_t == "K"
        rank = "King";
    end
    
    % Special care for rank 10
    if rank_t == "10"
        suit_head = bwareafilt(imcomplement(I_suitcrop),1);
        rank_head = bwareafilt(imcomplement(I_rankcrop),2);
    else
        suit_head = bwareafilt(imcomplement(I_suitcrop),1);
        rank_head = bwareafilt(imcomplement(I_rankcrop),1);
    end
    
    % Display all information
    figure(1)
    subplot(3,3,[1,2,4,5,7,8])
    imshow(I)
    xlabel('Input Image')
    title(rank + " of " + suit, 'FontSize', 18)
    subplot(3,3,3)
    imshow(Icrop)
    xlabel('Cropped Image')
    subplot(3,3,6)
    imshow((imcomplement(rank_head)))
    title(label_rank, 'FontSize', 18)
    subplot(3,3,9)
    imshow(imcomplement(suit_head))
    title(label_suit, 'FontSize', 18)
    pause;
end
