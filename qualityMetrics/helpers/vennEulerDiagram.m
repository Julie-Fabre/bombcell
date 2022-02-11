classdef vennEulerDiagram < matlab.graphics.chartcontainer.ChartContainer
    %vennEulerDiagram Create a Venn/Euler diagram for sets which can be area-proportional.
    %   vennEulerDiagram(setListData) create a Venn/Euler diagram which is
    %   area-proportional. setListData must be an n x 1 cell array of 
    %   vectors, where each vector corresponds to one of n sets. 
    %
    %   vennEulerDiagram(setMembershipData) create a Venn/Euler diagram which
    %   is area-proportional. setMembershipData must be an N x n logical
    %   matrix where the (i, j)th entry is 1 if element i is contained in
    %   set j.
    %
    %   vennEulerDiagram(setListData, setLabels) create a Venn/Euler 
    %   diagram which is area-proportional and has labels as specified in
    %   setLabels, an n x 1 string vector.
    %
    %   vennEulerDiagram(setMembershipData, setLabels) create a 
    %   Venn/Euler diagram which is area-proportional and has labels as 
    %   specified in setLabels, an n x 1 string vector. setMembershipData 
    %   must be an N x n logical matrix where the (i, j)th entry is 1 if 
    %   element i is contained in set j.
    %
    %   vennEulerDiagram() create an empty Venn/Euler diagram.
    %
    %   vennEulerDiagram(___,Name,Value) specifies additional options
    %   for the Venn/Euler chart using one or more name-value pair
    %   arguments. Specify the options after all other input arguments.
    %
    %   vennEulerDiagram(parent,___) creates the Venn/Euler chart in the 
    %   specified parent.
    %
    %   h = vennEulerDiagram(___) returns the Venn/Euler chart object.
    %   Use h to modify properties of the plot after creating it.
    
    %   Copyright 2021 The MathWorks, Inc.

    properties
        % N x n binary matrix indicating set membership
        SetMembershipData (:,:) logical = []

        % Labels for the sets
        SetLabels (:,1) string = []

        % Title of the venn/euler chart
        TitleText (:,1) string = []

        % Color order property used to draw diagram unless other color
        % properties are specified
        ColorOrder (:,3) {mustBeNumeric} = [];

        % Matrices where each row contains a 1x3 RGB vector denoting the
        % face colors and edge colors
        IntersectionColors (:,3) {mustBeNumeric} = []
        CircleFaceColors (:,3) {mustBeNumeric} = []
        CircleEdgeColors (:,3) {mustBeNumeric} = []

        % Column vectors where each row contains a value for face transparency
        % values and edge widths
        IntersectionTransparencies (:,1) {mustBeNumeric} = []
        CircleFaceTransparencies (:,1) {mustBeNumeric} = []
        CircleEdgeWidths (:,1) {mustBeNumeric} = []

        % Whether to show the number of elements in each disjoint intersection
        ShowIntersectionCounts (1,1) logical = false;

        % Whether to show the relative area of each disjoint intersection
        ShowIntersectionAreas (1,1) logical = false;
        
        % Whether the venn/euler diagram should be drawn with proportional
        % intersections
        DrawProportional (1,1) logical = true;
    end

    properties (SetAccess = private)
        % Read-only properties which are computed from the input data
        % Total number of sets in the input data
        NumSets (1,1) = 0

        % Vector containing the counts of elements in each disjoint
        % intersection of the sets
        SetCounts (:,1) {mustBeNumeric} = []

        % Vector containing the areas of the disjoint circle intersections
        IntersectionAreas (:,1) {mustBeNumeric} = []

        % Vector containing the x-data and y-data for circle centers
        CircleCenters (:,2) {mustBeNumeric} = []

        % Diameters of circles
        CircleDiameters (:,1) {mustBeNumeric} = []

        % Stress value (residual sum of squares divided by total sum of squares)
        Stress (1,1) = NaN
    end

    properties (Dependent)
        % Input data containing sets info
        % n x 1 cell array of sets (vectors of numbers or characters)
        SetListData (1,:) cell
    end

    properties (Access = protected)
        % number of sides for "circles"
        NumSides = 200;
    end

    properties(Access = private,Transient,NonCopyable)
        % Polygon array for storing circles
        Circles (:,1) matlab.graphics.primitive.Polygon

        % Polygon array for storing disjoint intersections
        DisjointIntersections (:,1) matlab.graphics.primitive.Polygon

        % Text labels
        SetLabelsText (:, 1) matlab.graphics.primitive.Text
        IntersectionText (:,1) matlab.graphics.primitive.Text
    end

    methods
        function obj = vennEulerDiagram(varargin)
            % Initialize list of arguments
            args = varargin;
            leadingArgs = cell(0);

            % Check if the first input argument is a graphics object to use as parent.
            if ~isempty(args) && isa(args{1},'matlab.graphics.Graphics')
                % vennEulerDiagram(parent, ___)
                leadingArgs = args(1);
                args = args(2:end);
            end

            % Check for optional positional arguments.
            if ~isempty(args)
                if mod(numel(args), 2) == 1
                    firstArg = args{1};

                    if iscell(firstArg)
                        % vennEulerDiagram(setListData)
                        % vennEulerDiagram(setListData,Name,Value)
                        setListData = firstArg;
                        leadingArgs = [leadingArgs {'SetListData', setListData}];
                        args = args(2:end);
                    else
                        % vennEulerDiagram(setMembershipData)
                        % vennEulerDiagram(setMembershipData,Name,Value)
                        setMembershipData = firstArg;
                        leadingArgs = [leadingArgs {'SetMembershipData', setMembershipData}];
                        args = args(2:end);
                    end

                elseif numel(args) >= 2
                    firstArg = args{1};

                    if iscell(firstArg)
                        % vennEulerDiagram(setListData,setLabels)
                        % vennEulerDiagram(setListData,setLabels,Name,Value)
                        setListData = firstArg;
                        setLabels = args{2};
                        leadingArgs = [leadingArgs {'SetListData', setListData, 'SetLabels', setLabels}];
                        args = args(3:end);
                    else
                        % vennEulerDiagram(setMembershipData, setLabels)
                        % vennEulerDiagram(setMembershipData,setLabels,Name,Value)
                        setMembershipData = firstArg;
                        setLabels = args{2};
                        leadingArgs = [leadingArgs {'SetMembershipData', setMembershipData, 'SetLabels', setLabels}];
                        args = args(3:end);
                    end
                end
            end

            % Combine positional arguments with name/value pairs.
            args = [leadingArgs args];

            % Call superclass constructor method
            obj@matlab.graphics.chartcontainer.ChartContainer(args{:});
        end

    end

    methods(Access = protected)
        function setup(obj)
            % Create the axes
            ax = getAxes(obj);

            % Adjust the aspect ratio of the axes to always be square
            ax.DataAspectRatio = [ 1 1 1 ];

            % Remove axes toolbar
            ax.Toolbar = [];

            % Remove appearance of axes ticks
            ax.XAxis.Visible = 'off';
            ax.YAxis.Visible  = 'off';

            % Remove datatips
            disableDefaultInteractivity(ax);
            
            % Set default colororder
            obj.ColorOrder = get(ax, 'ColorOrder');

            % For drawing the circles/disjoint intersections on the axes
            ax.NextPlot = 'add';
        end

        function update(obj)
            ax = getAxes(obj);

            % Remove empty sets from setMembershipData
            emptySetIndices = ~any(obj.SetMembershipData, 1);
            
            obj.SetMembershipData(:, emptySetIndices) = [];
            obj.NumSets = size(obj.SetMembershipData, 2);

            % Throw an error if the number of sets doesn't agree with the
            % number of setLabels
            if ~isempty(obj.SetLabels) && obj.NumSets ~= numel(obj.SetLabels)
                error("incorrectSize:SetLabels", "If specified, the number of set labels should be " + ...
                    "equal to the number of nonempty sets.")
            end

            % Identify duplicate sets in SetMembershipData, i.e. have columns
            % which are the same sequences of 0's and 1's
            [~, ia, ic] = unique(obj.SetMembershipData', 'rows', 'stable');
            uniqueSetLabels = [];
            uniqueSetMembershipData = obj.SetMembershipData(:, ia);
            obj.NumSets = size(uniqueSetMembershipData, 2);

            % Identify unique set labels
            if ~isempty(obj.SetLabels)
                uniqueSetLabels = strings(1, numel(ia));
    
                % Combine set labels for identical sets
                for uniqueIdx = 1:numel(ia)
                    % Set labels of the columns which share the same
                    % identifier
                    setLabels = obj.SetLabels(ic == uniqueIdx);
                    setLabel = strjoin(setLabels ,', ');
                    uniqueSetLabels(uniqueIdx) = setLabel;
                end
            end

            % Throw errors if any disjoint intersection graphics specifications are the
            % incorrect length.
            if ~isempty(obj.IntersectionColors) && size(obj.IntersectionColors, 1) ~= 2^obj.NumSets - 1
                error("incorrectSize:IntersectionColors", "If specified, disjoint intersection colors must be " + ...
                    "of length %d (2^n - 1, where n is the total number of nonempty sets).", 2^obj.NumSets - 1)
            elseif ~isempty(obj.IntersectionTransparencies) && numel(obj.IntersectionTransparencies) ~= 2^obj.NumSets - 1
                error("incorrectSize:IntersectionTransparencies", "If specified, disjoint intersection transparencies must be " + ...
                    "of length %d (2^n - 1, where n is the total number of nonempty sets).", 2^obj.NumSets - 1)
            end

            % Throw errors if any circle graphics specifications are the
            % incorrect length.
            if ~isempty(obj.CircleFaceColors) && size(obj.CircleFaceColors, 1) ~= obj.NumSets
                error("incorrectSize:CircleFaceColors", "If specified, circle face colors must be " + ...
                    "of length %d (the total number of nonempty sets).", obj.NumSets)
            elseif ~isempty(obj.CircleFaceTransparencies) && numel(obj.CircleFaceTransparencies) ~= obj.NumSets
                error("incorrectSize:CircleFaceTransparencies", "If specified, circle face transparencies must be " + ...
                    "of length %d (the total number of nonempty sets).", obj.NumSets)
            elseif ~isempty(obj.CircleEdgeColors) && size(obj.CircleEdgeColors, 1) ~= obj.NumSets
                error("incorrectSize:CircleEdgeColors", "If specified, circle edge colors must be " + ...
                    "of length %d (the total number of nonempty sets).", obj.NumSets)
            elseif ~isempty(obj.CircleEdgeWidths) && size(obj.CircleEdgeWidths, 1) ~= obj.NumSets
                error("incorrectSize:CircleEdgeWidths", "If specified, circle edge withs must be " + ...
                    "of length %d (the total number of nonempty sets).", obj.NumSets)
            end


            % Delete existing graphics objects
            delete(obj.DisjointIntersections);
            delete(obj.Circles);
            delete(obj.IntersectionText);
            delete(obj.SetLabelsText);

            % If the set data are empty, return since nothing needs to be
            % added to the chart
            if isempty(obj.SetMembershipData)
                return
            end

            % If a non-proportional venn diagram is being drawn, there can
            % be at most three sets
            if ~obj.DrawProportional && obj.NumSets > 3
                error(['vennEulerDiagram: A non-proportional vennEulerDiagram can ' ...
                    'only be created for up to three sets.'])
            end

            % Create the counts vector and the area vector, which are of
            % length 2^n
            obj.SetCounts = zeros(2^obj.NumSets, 1);
            obj.IntersectionAreas = zeros(2^obj.NumSets, 1);

            % Compute the counts of each disjoint intersection
            for rowIdx = 1:size(uniqueSetMembershipData, 1)
                currRow = uniqueSetMembershipData(rowIdx, :);

                % Convert binary vector to integer. Update the set count
                % for the corresponding set
                binaryVector = flip(currRow);
                binaryString = sprintf('%d', binaryVector);
                countsIdx = bin2dec(binaryString);
                obj.SetCounts(countsIdx + 1) = obj.SetCounts(countsIdx + 1) + 1;
            end

            % If not drawing a proportional diagram, simply set centers to
            % be one point, two same-y points, or three equally spaced
            % points.
            if obj.DrawProportional && obj.NumSets > 1
                % Create an Euler diagram
                normalizedSetCounts = obj.SetCounts / sum(obj.SetCounts);
                [obj.Stress, obj.CircleDiameters, obj.CircleCenters, circlePolyshapes, ...
                    obj.IntersectionAreas, intersectionPolyshapes, shownIntersectionIndices] = ...
                    vennEulerDiagram.computeCircleInfo(normalizedSetCounts, obj.NumSets, obj.NumSides);
            
            else
                % Create a pure venn diagram
                [obj.CircleDiameters, obj.CircleCenters] = vennEulerDiagram.fixedCircleInfo(obj.NumSets);

                % Create all the circles using the centers and diameters.
                circlePolyshapes = [];

                for setIdx = 1:obj.NumSets
                    centerCoords = obj.CircleCenters(setIdx, :);
                    radius = obj.CircleDiameters(setIdx) / 2;
        
                    circlePolyshapes = [circlePolyshapes, nsidedpoly(obj.NumSides, 'Center', centerCoords, 'Radius', radius)]; %#ok<AGROW> 
                end
        
                % Get disjoint intersection areas and handles
                [areas, intersectionPolyshapes] = vennEulerDiagram.findDisjointIntersectionAreas(obj.NumSets, circlePolyshapes);

                % Update areas vector
                obj.IntersectionAreas = areas;
            end

            % Determine whether to draw disjoint intersections or circles
            if ~isempty(obj.IntersectionColors)
                polygons = plot(ax, intersectionPolyshapes);

                colors = obj.IntersectionColors;
                transparencies = obj.IntersectionTransparencies;
            else
                polygons = plot(ax, circlePolyshapes);

                colors = obj.CircleFaceColors;
                transparencies = obj.CircleFaceTransparencies;

                % Apply edge colors if nonempty, otherwise save defaults in
                % corresponding property
                if ~isempty(obj.CircleEdgeColors)
                    for i = 1:obj.NumSets
                        polygons(i).EdgeColor = obj.CircleEdgeColors(i, :);
                    end
                else
                    obj.CircleEdgeColors = vertcat(polygons.EdgeColor);
                end

                % Apply edge line widths if nonempty, otherwise save
                % defaults in corresponding property
                if ~isempty(obj.CircleEdgeWidths)
                    for i = 1:obj.NumSets
                        polygons(i).LineWidth = obj.CircleEdgeWidths(i);
                    end
                else
                    obj.CircleEdgeWidths = vertcat(polygons.LineWidth);
                end
            end

            % Apply face colors if nonempty
            if ~isempty(colors)
                for i = 1:numel(polygons)
                    polygons(i).FaceColor = colors(i, :);
                end
            % Otherwise apply colororder colors and save face colors in
            % corresponding property
            else
                numColors = size(obj.ColorOrder,1);
                faceColors = ones(numel(polygons), 3);

                for i = 1:numel(polygons)
                    colorIndex = mod(i - 1, numColors) + 1;
                    color = obj.ColorOrder(colorIndex, :);
                    polygons(i).FaceColor = color;

                    faceColors(i, :) = color;
                end

                if ~isempty(obj.IntersectionColors)
                    obj.IntersectionColors = faceColors;
                else
                    obj.CircleFaceColors = faceColors;
                end
            end

            % Apply face transparencies if nonempty
            if ~isempty(transparencies)
                for i = 1:numel(polygons)
                    polygons(i).FaceAlpha = transparencies(i);
                end
            % Otherwise save default transparencies in corresponding
            % property
            else
                faceTransparencies = vertcat(polygons.FaceAlpha);

                if ~isempty(obj.IntersectionColors)
                    obj.IntersectionTransparencies = faceTransparencies;
                else
                    obj.CircleFaceTransparencies = faceTransparencies;
                end
            end

            % Save polygons
            if ~isempty(obj.IntersectionColors)
                obj.DisjointIntersections = polygons;
            else
                obj.Circles = polygons;
            end

            % Create set labels
            if ~isempty(uniqueSetLabels)
                setLabelsText = gobjects(numel(uniqueSetLabels), 1);
    
                % Label each of the areas inside the venn diagram
                for setIdx = 1:numel(uniqueSetLabels)
                    centerCoords = obj.CircleCenters(setIdx, :);
                    setLabel = uniqueSetLabels(setIdx);
    
                    setLabelsText(setIdx) = text(ax, centerCoords(1), centerCoords(2), setLabel, ...
                        'HorizontalAlignment', 'center');
                end
               
                obj.SetLabelsText = setLabelsText;
            end

            % Show number of elements (counts) of each disjoint
            % intersection and/or relative areas of each disjoint
            % intersection
            if obj.ShowIntersectionCounts || obj.ShowIntersectionAreas
                % Find the mean point of each disjoint intersection
                meanPts = zeros(2^obj.NumSets, 2);

                for intersectionIdx = 1:numel(intersectionPolyshapes)
                    vertexCoordinates = intersectionPolyshapes(intersectionIdx).Vertices;
                    NaNIndices = any(isnan(vertexCoordinates), 2);
                    vertexCoordinates(NaNIndices, :) = [];
                    meanPts(intersectionIdx, :) = mean(vertexCoordinates, 1);
                end

                % If drawing a pure venn diagram, all disjoint
                % intersections are shown.
                if ~obj.DrawProportional
                    shownIntersectionIndices = 2:2^obj.NumSets;
                end

                % Populate textLabels with intersection counts and areas
                textLabels = [];
                if obj.ShowIntersectionCounts
                    textLabels = string(obj.SetCounts(shownIntersectionIndices));
                end
                if obj.ShowIntersectionAreas
                    relativeAreas = obj.IntersectionAreas(shownIntersectionIndices) / 100;
                    textLabels = [textLabels, string(round(relativeAreas, 3))];
                end

                % Save handles of text objects
                diagramText = gobjects(numel(intersectionPolyshapes), 1);
                
                for intersectionIdx = 1:numel(intersectionPolyshapes)
                    coordinate = meanPts(intersectionIdx, :);
                    diagramText(intersectionIdx) = text(ax, coordinate(1), coordinate(2), ...
                        textLabels(intersectionIdx, :), ...
                        'HorizontalAlignment', 'center');
                end

                obj.IntersectionText = diagramText;
            end

            % Set title
            title(getAxes(obj), obj.TitleText);
        end

        function groups = getPropertyGroups(obj)
            if ~isscalar(obj)
                % List for array of objects
                % Can call from immediate superclass
                groups = getPropertyGroups@matlab.mixin.CustomDisplay(obj);
            else
                % List for scalar object
                propList = cell(1,0);
                
                % Add the title, if not empty or missing.
                nonEmptyTitle = obj.TitleText ~= "" & ~ismissing(obj.TitleText);
                if any(nonEmptyTitle)
                    propList = {'TitleText'};
                end
                
                % Add properties which are not read-only.
                % Check that stylistic properties are non-empty before
                % adding.
                propList{end+1} = 'SetMembershipData';
                propList{end+1} = 'SetLabels';

                if ~isempty(obj.IntersectionColors)
                    propList{end+1} = 'IntersectionColors';
                end

                if ~isempty(obj.CircleFaceColors)
                    propList{end+1} = 'CircleFaceColors';
                end

                if ~isempty(obj.CircleEdgeColors)
                    propList{end+1} = 'CircleEdgeColors';
                end

                if ~isempty(obj.CircleFaceTransparencies)
                    propList{end+1} = 'CircleFaceTransparencies';
                end

                if ~isempty(obj.CircleEdgeWidths)
                    propList{end+1} = 'CircleEdgeWidths';
                end

                propList{end+1} = 'ShowIntersectionCounts';
                propList{end+1} = 'ShowIntersectionAreas';
                propList{end+1} = 'DrawProportional';
                
                groups = matlab.mixin.util.PropertyGroup(propList);
            end
        end

    end

    methods (Static)
        function [circleDiameters, circleCenters] = fixedCircleInfo(numSets)
        % fixedCircleInfo  Compute the circle diameters and circle centers for an
        % unproportional venn diagram (determinstic locations and diameters).
        %   [circleDiameters, circleCenters] = fixedCircleInfo(numSets)
        %   computes the circle diameters and circle centers for numSets sets
        %   in an unproportional venn diagram.
        %
        %   See also computeCircleInfo.

            circleDiameters = 2 * ones(numSets, 1);
            circleCenters = zeros(numSets, 2);

            % For n = 1, 2, or 3 sets, create equally spaced center points
            % and equal diameters.
            switch numSets
                case 1
                    circleCenters(1, :) = [0 0];
                case 2
                    distFromOrigin = 0.7;

                    circleCenters(1, :) = [-distFromOrigin, 0];
                    circleCenters(2, :) = [distFromOrigin, 0];
                case 3
                    distFromOrigin = 0.7;
                    
                    circleCenters(1, :) = [distFromOrigin * cos(5*pi/6), distFromOrigin * sin(pi/6)];
                    circleCenters(2, :) = [distFromOrigin * cos(3*pi/2), distFromOrigin * sin(3*pi/2)];
                    circleCenters(3, :) = [distFromOrigin * cos(pi/6), distFromOrigin * sin(pi/6)];
            end
        end

        function [stress, circleDiameters, circleCenters, circlePolyshapes, ...
                intersectionAreas, intersectionPolyshapes, shownIntersectionIndices] = ...
                computeCircleInfo(setCounts, numSets, numSides)
        % computeCircleInfo  Compute values necessary for plotting a
        % proportional venn diagram using the venneuler() algorithm.
        %   [stress, circleDiameters, circleCenters, circleHandles, intersectionAreas, intersectionHandles] = ...
        %       computeCircleInfo(setCounts, numSets, numSides)
        %   computes:
        %       stress -- stress value (residual sum of squares / total sum 
        %       of squares for setCounts and circle areas)
        %       circleDiameters -- circle diameters for proportional diagram
        %       circleCenters -- circle centers for proportional diagram
        %       circleHandles -- polyshape handles for circles
        %       intersectionAreas -- percentage areas for disjoint
        %       intersections of sets
        %       intersectionPolyshapes -- polyshapes array for disjoint
        %       intersections of sets
        %       shownIntersectionIndices -- indices of disjoint
        %       intersections which are shown
        %   given:
        %       setCounts -- relative counts of each disjoint intersection
        %       numSets -- total number of sets 
        %       numSides -- number of sides used for each polyshape
        %       approximating a circle
        %
        %   See also fixedCircleInfo.

            circleAreas = vennEulerDiagram.computeCircleAreas(numSets, setCounts);
            initialCenters = vennEulerDiagram.computeInitialCircleCenters(setCounts, numSets, circleAreas);
            circleDiameters = vennEulerDiagram.computeCircleDiameters(numSets, circleAreas);

            % Iterations proceed by holding diameters fixed and moving the
            % circle centers.
            [stress, circleCenters, circlePolyshapes, intersectionAreas, intersectionPolyshapes, ...
                shownIntersectionIndices] = ...
                vennEulerDiagram.minimizeLoss(setCounts, numSets, numSides, initialCenters, circleDiameters);
        end

        function circleAreas = computeCircleAreas(numSets, setCounts)
        % computeCircleAreas  Compute the areas of circles.
        %   circleAreas = computeCircleAreas(numSets, setCounts) computes
        %   the relative areas of numSets circles given setCounts, a vector 
        %   of the relative counts of each disjoint intersection.

            circleAreas = zeros(numSets, 1);

            for setCountsIdx = 1:numel(setCounts)
                % If this disjoint intersection is nonempty
                if setCounts(setCountsIdx) 
                    % Convert index to binary vector
                    paddedBinaryVec = decimalToBinaryVector(setCountsIdx - 1, numSets);
    
                    % For each location where there is a 1, add the value
                    % to the corresponding class
                    for setIdx = 1:numSets
                        if paddedBinaryVec(setIdx)
                            circleAreas(setIdx) = circleAreas(setIdx) + setCounts(setCountsIdx);
                        end

                    end
                end
            end
        end

        function [initialCenters] = computeInitialCircleCenters(setCounts, numSets, circleAreas)
        % computeCircleAreas  Compute the initial circle locations using an approach 
        % from multidimensional scaling. This reduces the likelihood of falling 
        % into local minima.
        %   [initialCenters] = computeInitialCircleCenters(setCounts, numSets, circleAreas) 
        %   computes:
        %       initialCenters -- a numSets x 2 matrix where the first
        %       column contains the x-values of the centers and
        %       the second column contains the y-values of the centers
        %   given:
        %       setCounts -- relative counts of each disjoint intersection
        %       numSets -- total number of sets
        %       circleAreas -- relative area of each circle

            % Compute a Jaccard distance matrix |intersection|/|union|
            distances = zeros(numSets, numSets);
            for i = 1:numSets
                for j = i + 1:numSets
                    % Convert to integer to get index for counts
                    intersectionCount = 0;

                    % Go through each possible disjoint intersection
                    for setCountsIdx = 1:numel(setCounts)
                        if setCounts(setCountsIdx)

                            % Mapping from binary indices from classes is
                            % from right to left, so use right-msb
                            paddedBinaryVec = decimalToBinaryVector(setCountsIdx - 1, numSets);
    
                            % If this disjoint intersection is included in both
                            % sets, add the counts to the intersection count
                            if paddedBinaryVec(i) && paddedBinaryVec(j)
                                intersectionCount = intersectionCount + setCounts(setCountsIdx);
                            end
                        end
                    end

                    distances(i, j) = 1 - intersectionCount / (circleAreas(i) + circleAreas(j));
                    distances(j, i) = distances(i, j);
                end
            end

            % Square the distances to set up the squared proximity matrix.
            % Also multiply by a factor -0.5 to apply double centering
            distances = -0.5 * distances.^2;

            % Normalize the distance matrix to have 0 mean for each row and
            % for each column
            rowMeans = mean(distances, 2);
            colMeans = mean(distances, 1);
            matrixMean = mean(distances, 'all');

            distances = distances + matrixMean;
            distances = distances - rowMeans;
            distances = distances - colMeans;

            % Use SVD and eigenvalues to compute initial centers
            [U, S, ~] = svd(distances);
            U = U(:, 1:2);
            eigenValues = S(1:2, 1:2);
            initialCenters = U * sqrt(eigenValues);

            % Center the circle centers at the origin
            initialCenters = initialCenters - mean(initialCenters);

            % Scale centers to have an average distance to the origin of
            % 0.1
            goalAvgDistanceToOrigin = 0.1;

            avgDistanceToOrigin = 0;
            for i = 1:numel(initialCenters)
                avgDistanceToOrigin = avgDistanceToOrigin + norm(initialCenters(i));
            end

            avgDistanceToOrigin = avgDistanceToOrigin / numel(initialCenters);
            initialCenters = initialCenters * goalAvgDistanceToOrigin / avgDistanceToOrigin;
        end

        function [circleDiameters] = computeCircleDiameters(numSets, circleAreas)
        % computeCircleDiameters  Given numSets, the total number of sets, and
        % circleAreas, the relative areas of each circle, compute the
        % scaled circle diameters.
            circleDiameters = 2 * sqrt(circleAreas / (pi * numSets));
        end

        function [finalStress, finalCircleCenters, circlePolyshapes, intersectionAreas, intersectionPolyshapes, ...
                shownIntersectionIndices] ...
                = minimizeLoss(setCounts, numSets, numSides, initialCenters, initialDiameters)
        % minimizeLoss  Compute final values necessary for plotting a
        % proportional venn diagram by approximately minimizing OLS loss.
        %   finalStress, finalCircleCenters, circleHandles, intersectionAreas, intersectionPolyshapes] ...
        %        = minimizeLoss(setCounts, numSets, numSides, initialCenters, initialDiameters)
        %   computes:
        %       finalStress -- final stress value (residual sum of squares / total sum 
        %       of squares for setCounts and circle areas)
        %       finalCircleCenters -- final circle centers for proportional diagram
        %       circleHandles -- polyshape handles for circles
        %       intersectionAreas -- percentage areas for disjoint
        %       intersections of sets
        %       intersectionPolyshapes -- polyshapes array for disjoint
        %       intersections of sets
        %       shownIntersectionIndices -- indices of disjoint
        %       intersections which are shown
        %   given:
        %       setCounts -- relative counts of each disjoint intersection
        %       numSets -- total number of sets 
        %       numSides -- number of sides used for each polyshape
        %       approximating a circle
        %       initialCenters -- initial centers of the circles
        %       initialDiameters -- initial diameters of the circles
            
            % Difference threshold and minimum stress (stopping conditions)
            epsilon = 0.000001;

            % Number of iterations for gradient descent
            numIterations = 100;
            minIterations = 10;

            % Centers for the previous/current iteration
            previousCenters = initialCenters;
            currentCenters = previousCenters;

            % Stress value for last iteration
            currentStress = 0.5;
            previousStress = 1;
            
            for iter = 1:numIterations
                % Recenter the center coordinates to be about the origin
                initialCenters = initialCenters - mean(initialCenters);
                
                % Get handles for polyshapes representing circles
                circlePolyshapes = vennEulerDiagram.getCirclePolyshapes(numSets, currentCenters, initialDiameters, numSides);
    
                % Compute the actual areas of each disjoint intersection
                % based on the current centers
                [intersectionAreas, intersectionPolyshapes, shownIntersectionIndices] = ...
                    vennEulerDiagram.findDisjointIntersectionAreas(numSets, circlePolyshapes);

                % Compute stress (SSE/SST) and area estimates using OLS method
                [currentStress, areas_hat] = vennEulerDiagram.computeStress(intersectionAreas, setCounts);

                % If the new stress value for the current centers is greater
                % than it was for the previous centers, revert back to the
                % previous centers. Otherwise update the previous centers
                if currentStress > previousStress
                    currentCenters = previousCenters;
                else
                    previousCenters = currentCenters;
                end

                % Check stopping conditions
                if iter > minIterations && (currentStress < epsilon || (previousStress - currentStress) < epsilon)
                    break;
                end

                % Compute gradients and update centers
                gradients = vennEulerDiagram.computeGradients(numSets, currentCenters, intersectionAreas, areas_hat);
                currentCenters = previousCenters + gradients;
                previousStress = currentStress;
            end
    
            finalCircleCenters = currentCenters;
            finalStress = currentStress;
            
        end

        function [gradients] = computeGradients(numSets, currCircleCenters, areas, areas_hat)
        % computeGradients  Compute gradients for a single iteration of
        % gradient descent.
        %   [gradients] = computeGradients(numSets, currCenters, areas, areas_hat)
        %   computes:
        %       gradients -- a numSets x 2 matrix where the first
        %       column contains the x-components of the gradients and
        %       the second column contains the y-components
        %   given:
        %       numSets -- total number of sets 
        %       currCircleCenters -- current centers of the circles
        %       areas -- percentage areas for disjoint intersections of sets
        %       areas_hat -- OLS estimate of areas of disjoint
        %       intersections

            % Define step size for gradient 
            stepSize = 0.01;
            
            % Create and populate gradients
            gradients = zeros(numSets, 2);

            for k = 1:2^numSets
                paddedBinaryVec = decimalToBinaryVector(k - 1, numSets);

                for i = 1:numSets
                    for j = i+1:numSets
                        if paddedBinaryVec(i) && paddedBinaryVec(j)
                            gradients(i, 1) = gradients(i, 1) + stepSize * (currCircleCenters(i, 1) - currCircleCenters(j, 1)) * ...
                                (areas(k) - areas_hat(k));
                            gradients(i, 2) = gradients(i, 2) + stepSize * (currCircleCenters(i, 2) - currCircleCenters(j, 2)) * ...
                                (areas(k) - areas_hat(k));
                            gradients(j, 1) = gradients(j, 1) - stepSize * (currCircleCenters(i, 1) - currCircleCenters(j, 1)) * ...
                                (areas(k) - areas_hat(k));
                            gradients(j, 2) = gradients(j, 2) - stepSize * (currCircleCenters(i, 2) - currCircleCenters(j, 2)) * ...
                                (areas(k) - areas_hat(k));
                        end
                    end
                end
            end
        end

        function [circlePolyshapes] = getCirclePolyshapes(numSets, circleCenters, circleDiameters, numSides)
        % getCirclePolyshapes  Create polyshapes for circles.
        %   [newCircles] = getCirclePolyshapes(numSets, currCenters, circleDiameters, numSides)
        %   computes:
        %       circlePolyshapes-- polyshapes corresponding to each circle
        %   given:
        %       numSets -- total number of sets 
        %       circleCenters -- two-column matrix of centers of the circles
        %       circleDiameters -- diameters of the circles
        %       numSides -- number of sides used for each polyshape
        %       approximating a circle

            circlePolyshapes = [];
    
            % Create all the circles using the centers and diameters.
            for setIdx = 1:numSets
                centerCoords = circleCenters(setIdx, :);
                radius = circleDiameters(setIdx) / 2;

                circlePolyshapes = [circlePolyshapes, nsidedpoly(numSides, 'Center', centerCoords, 'Radius', radius)]; %#ok<AGROW> 
            end
        end

        function [stress, areas_hat] = computeStress(intersectionAreas, setCounts)
        % computeStress Compute the stress value associated with a venn/euler diagram.
        %   [stress, areas_hat] = computeStress(areas, counts)
        %   computes:
        %       stress -- residual sum of squares / total sum of squares
        %   given:
        %       intersectionAreas -- percentage areas for disjoint
        %       intersections of sets
        %       setCounts -- relative counts of each disjoint intersection       

            % OLS estimate for proportionality constant
            beta_hat = intersectionAreas' * setCounts / (setCounts' * setCounts);

            % Residual sum of squares and total sum of squares
            SSE = (intersectionAreas - beta_hat * setCounts)' * (intersectionAreas - beta_hat * setCounts);
            SST = intersectionAreas' * intersectionAreas;

            stress = SSE / SST;
            areas_hat = beta_hat * setCounts;
        end

        function [intersectionAreas, intersectionPolyshapes, shownIntersectionIndices] = findDisjointIntersectionAreas(numSets, circlePolyshapes)  
        % findDisjointIntersectionAreas  Create the polyshapes for the
        % disjoint intersections of the sets.
        %   [areas, areaHandles] = findDisjointIntersectionAreas(numSets, circleHandles)
        %   computes:
        %       intersectionAreas -- percentage areas for disjoint
        %       intersections of sets
        %       intersectionPolyshapes -- polyshapes array for disjoint
        %       intersections of sets
        %       shownIntersectionIndices -- indices of disjoint
        %       intersections which are shown
        %   given:
        %       numSets -- total number of sets 
        %       circlePolyshapes-- polyshapes corresponding to each circle

            % Vector of areas of each disjoint intersection
            intersectionAreas = zeros(2^numSets, 1);
            intersectionPolyshapes = [];
            shownIntersectionIndices = [];

            for setCountsIdx = 1:2^numSets
                paddedBinaryVec = decimalToBinaryVector(setCountsIdx - 1, numSets);
                
                % Get the intersection of the sets
                includedSets = circlePolyshapes(logical(paddedBinaryVec)); 
                excludedSets = circlePolyshapes(~logical(paddedBinaryVec));

                areaHandle = polyshape();
                
                if numel(includedSets) >= 1
                    % Intersect all the included sets
                    areaHandle = intersect(includedSets); %#ok<LTARG> 

                    % For each excluded set, intersect it with the current 
                    % set, then subtract the area using xor
                    for excludedSet = excludedSets
                        subset = intersect(excludedSet, areaHandle);
                        areaHandle = xor(areaHandle, subset);
                    end
                end

                intersectionAreas(setCountsIdx) = area(areaHandle);

                if numel(areaHandle.Vertices)
                    shownIntersectionIndices(end + 1) = setCountsIdx; %#ok<AGROW> 
                    intersectionPolyshapes = [intersectionPolyshapes, areaHandle]; %#ok<AGROW> 
                end
            end

            % Convert areas to percentages
            intersectionAreas = 100 * intersectionAreas / sum(intersectionAreas);
        end

    end

    methods
        % title method
        function title(obj,txt)
            if nargin>=2
                if isnumeric(txt)
                    txt=num2str(txt);
                end
                obj.TitleText = txt;
            end
        end
    end

    methods
        % Set method for creating SetMembershipData from SetListData
        function set.SetListData(obj, setListData)

            numSets = numel(setListData);
            
            % Go through the vectors, get unique elements and map them
            % to the row number in setMembershipData.
            kType = 'char';
            ElementsToRowIdx = containers.Map('KeyType', kType, 'ValueType', 'double');
            currRow = 1;

            for setIdx = 1:numel(setListData)
                setVector = setListData{setIdx};
                
                for elemIdx = 1:numel(setVector)
                    currElem = setVector(elemIdx);
                    currElem = char(string(currElem));

                    % If the current element hasn't been recorded as a
                    % unique element and mapped to a membership row, add
                    % mapping
                    if ~isKey(ElementsToRowIdx, currElem)
                        ElementsToRowIdx(currElem) = currRow;
                        currRow = currRow + 1;
                    end
                end
            end

            % Create setMembership to be N x n (elements by number of
            % sets)
            setMembershipData = zeros(numel(keys(ElementsToRowIdx)), numSets);

            % Go through each vector and update setMembership Data
            % according to the ElementsToRowIdx map
            for setIdx = 1:numel(setListData)
                setVector = setListData{setIdx};

                for elemIdx = 1:numel(setVector)
                    currElem = setVector(elemIdx);
                    currElem = char(string(currElem));
                    rowIdx = ElementsToRowIdx(currElem);

                    setMembershipData(rowIdx, setIdx) = 1;
                end
            end

            obj.SetMembershipData = setMembershipData;
        end

        % Get method for getting SetListData from SetMembershipData when
        % queried
        function setListData = get.SetListData(obj)

            numSets = size(obj.SetMembershipData, 2);
            setListData = cell(numSets, 1);

            % Sequential list of possible elements
            elements = 1:size(obj.SetMembershipData, 1);
            elements = elements(:);

            % From each column in SetMembership, we can identify the
            % elements (indices where there is a 1).
            for setIdx = 1:numSets
                binaryElementsVector = logical(obj.SetMembershipData(:, setIdx));
                currentSet = elements(binaryElementsVector);

                setListData{setIdx, 1} = currentSet(:)';
            end
        end

    end

end

% Helper function for converting decimal values to binary vectors, where
% the leftmost element has the lowest place value
function [paddedBinaryVector] = decimalToBinaryVector(decimalValue, numSets)
    binaryString = flip(dec2bin(decimalValue));
    binaryVector = binaryString - '0';

    paddedBinaryVector = [binaryVector, zeros(1, numSets - numel(binaryVector))];
end