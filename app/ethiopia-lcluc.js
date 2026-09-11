///////////////////////////////////////////////////////////////
// Ethiopia / Amhara Land Cover Explorer
///////////////////////////////////////////////////////////////

// -------------------------
// 1) Inputs
// -------------------------
var ethiopiaBoundary = ee.FeatureCollection(
  'projects/ee-jacaraba-ethiopia/assets/boundaries/Amhara_Study_Area_Boundary_4buf10km_EPSG_GEE'
);

// Aligned evaluation products already uploaded to the GSFC DSG project.
// Select the first band because uploaded GeoTIFF band names may vary.
function comparisonImage(assetName) {
  return ee.Image('projects/gsfc-dsg/assets/' + assetName)
    .select([0])
    .clip(ethiopiaBoundary);
}

var deaCropland = comparisonImage(
  'DigitalEarthAfrica_crop_mask_2019_Amhara_LCLUcrop0_nonCrop255'
);
// Crop is 0. Preserve fractional coverage in the asset's overview mask:
// replacing it with eq(0) alone can make sparse crop coverage fully opaque.
// Non-crop codes (8 or 255, depending on the source) remain hidden.
deaCropland = deaCropland.updateMask(
  deaCropland.mask().multiply(deaCropland.eq(0))
);

var worldCover2020 = comparisonImage(
  'ESA_WorldCover_10m_2020_v100_Amhara_reclass'
);
var esri2020 = comparisonImage('ESRI_LULC_36P37P_2020_Amhara_reclass');
var glad2020 = comparisonImage('GLAD2020_Amhara_reclass');
var dynamicWorld2020 = comparisonImage(
  'Google_DynamicWorld_LULC_2020_mode_2_reclass'
);

var validationPoints = ee.FeatureCollection(
  'projects/gsfc-dsg/assets/Amhara_validation_points_2026'
).filterBounds(ethiopiaBoundary);

var canopyHeight = ee.ImageCollection(
  'projects/sat-io/open-datasets/facebook/meta-canopy-height'
).mosaic().clip(ethiopiaBoundary);


// GSFC DSG assets
var lc_2009_2016 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_LCLU_5class_2009_2016_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var lc_2018_2022 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_LCLU_5class_2018_2022_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var lc_2017_2024 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_LCLU_5class_2017_2024_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var nobs_2009_2016 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_nobservations_2009_2016_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var nobs_2018_2022 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_nobservations_2018_2022_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var nobs_2017_2024 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_nobservations_2017_2024_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);


// -------------------------
// 2) Visualization
// -------------------------
var palettes = require('users/gena/packages:palettes');

var lcPalette = [
  '#ffaa00', // Crop
  '#267300', // Tree/Shrub
  '#ffffbe', // Grass
  '#730000', // Built
  '#0070ff'  // Water
];

var lcNames = [
  'Crop',
  'Tree / Shrub',
  'Grass',
  'Built',
  'Water'
];

// Local raster values and validation-point sampling support codes 0–4 above.
// See README.md for the class scheme and limits of this verification.
var lcVis = {
  min: 0,
  max: 4,
  palette: lcPalette
};

// Explicitly mask each GeoTIFF's NoData value, preserving valid class 0.
function classifiedComparison(image, noData) {
  return image.updateMask(image.neq(noData));
}

// Treat ESRI code 7 as NoData along with its declared NoData value 15.
var esriDisplay = classifiedComparison(esri2020, 15)
  .updateMask(esri2020.neq(7));

var validationStyle = {
  color: '000000',
  fillColor: 'ff00ff',
  pointSize: 5,
  width: 1
};

var nobsVis = {
  min: 0,
  max: 100,
  palette: palettes.matplotlib.magma[7]
};

var chmVis = {
  min: 0,
  max: 20,
  palette: palettes.matplotlib.viridis[7]
};


// -------------------------
// 3) Map setup
// -------------------------
Map.setOptions('SATELLITE');
Map.centerObject(ethiopiaBoundary, 7);
Map.style().set('cursor', 'crosshair');

var boundaryLayer = ui.Map.Layer(
  ethiopiaBoundary.style({
    color: 'ffffff',
    fillColor: '00000000',
    width: 2
  }),
  {},
  'Amhara Study Boundary',
  true
);

Map.add(boundaryLayer);


// -------------------------
// 4) Add map layers
// -------------------------
function addLayer(image, vis, name, shown, opacity) {
  var layer = ui.Map.Layer(image, vis, name, shown, opacity || 1.0);
  Map.add(layer);
  return layer;
}

var layers = {
  dea: addLayer(
    deaCropland,
    {min: 0, max: 1, palette: [lcPalette[0]]},
    'Digital Earth Africa Cropland 2019 (aligned)',
    false
  ),

  worldCov: addLayer(
    classifiedComparison(worldCover2020, -128),
    lcVis,
    'ESA WorldCover 2020 (aligned/reclassified)',
    false
  ),

  esri: addLayer(
    esriDisplay,
    lcVis,
    'ESRI Land Cover 2020 (aligned/reclassified)',
    false
  ),

  glad: addLayer(
    classifiedComparison(glad2020, 15),
    lcVis,
    'GLAD 2020 (aligned/reclassified)',
    false
  ),

  dynamicWorld: addLayer(
    classifiedComparison(dynamicWorld2020, 15),
    lcVis,
    'Google Dynamic World 2020 (aligned/reclassified)',
    false
  ),

  chm: addLayer(
    canopyHeight,
    chmVis,
    'Meta Canopy Height 1 m',
    false
  ),

  lc0916: addLayer(
    lc_2009_2016,
    lcVis,
    'GSFC LCLU 2 m: 2009–2016',
    false
  ),

  lc1822: addLayer(
    lc_2018_2022,
    lcVis,
    'GSFC LCLU 2 m: 2018–2022',
    false
  ),

  lc1724: addLayer(
    lc_2017_2024,
    lcVis,
    'GSFC LCLU 2 m: 2017–2024',
    true
  ),

  nobs0916: addLayer(
    nobs_2009_2016,
    nobsVis,
    'Observations: 2009–2016',
    false,
    0.85
  ),

  nobs1822: addLayer(
    nobs_2018_2022,
    nobsVis,
    'Observations: 2018–2022',
    false,
    0.85
  ),

  nobs1724: addLayer(
    nobs_2017_2024,
    nobsVis,
    'Observations: 2017–2024',
    false,
    0.85
  )
};

// Add points last so they remain visible above raster layers when enabled.
layers.validation = addLayer(
  validationPoints.style(validationStyle),
  {},
  'Validation / Reference Points 2026',
  false
);


// -------------------------
// 5) UI helpers
// -------------------------
function titleLabel(text) {
  return ui.Label(text, {
    fontSize: '22px',
    fontWeight: 'bold',
    color: '#4A997E',
    margin: '8px 8px 4px 8px'
  });
}

function subtitleLabel(text) {
  return ui.Label(text, {
    fontSize: '13px',
    color: '#555555',
    margin: '0 8px 12px 8px'
  });
}

function sectionLabel(text) {
  return ui.Label(text, {
    fontWeight: 'bold',
    fontSize: '15px',
    color: '#2F6F5E',
    margin: '14px 8px 6px 8px'
  });
}

function makeCheckbox(label, layer, defaultValue) {
  var checkbox = ui.Checkbox({
    label: label,
    value: defaultValue,
    style: {margin: '2px 8px'}
  });

  checkbox.onChange(function(checked) {
    layer.setShown(checked);
    if (layer === layers.validation && !checked) {
      closeValidationPopup();
    }
  });

  return checkbox;
}

function makeOpacitySlider(layer, label, defaultValue) {
  var slider = ui.Slider({
    min: 0,
    max: 1,
    value: defaultValue,
    step: 0.05,
    style: {
      stretch: 'horizontal',
      margin: '0 8px 8px 8px'
    }
  });

  slider.onChange(function(value) {
    layer.setOpacity(value);
  });

  return ui.Panel([
    ui.Label(label, {
      fontSize: '11px',
      color: '#666666',
      margin: '4px 8px 0 8px'
    }),
    slider
  ]);
}

function makeLegendRow(color, name) {
  return ui.Panel({
    widgets: [
      ui.Label({
        style: {
          backgroundColor: color,
          padding: '8px',
          margin: '0 0 4px 0'
        }
      }),
      ui.Label({
        value: name,
        style: {
          margin: '0 0 4px 6px',
          fontSize: '12px'
        }
      })
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
}

function makeDiscreteLegend(title, colors, names, description) {
  var legend = ui.Panel({
    style: {
      margin: '8px',
      padding: '8px',
      backgroundColor: 'rgba(255,255,255,0.90)',
      border: '1px solid #dddddd'
    }
  });

  legend.add(ui.Label(title, {
    fontWeight: 'bold',
    fontSize: '13px',
    color: '#333333',
    margin: '0 0 6px 0'
  }));

  if (description) {
    legend.add(ui.Label(description, {
      fontSize: '11px', color: '#555555', margin: '0 0 8px 0'
    }));
  }

  for (var i = 0; i < colors.length; i++) {
    legend.add(makeLegendRow(colors[i], names[i]));
  }

  return legend;
}

function makeGradientLegend(title, vis, leftLabel, rightLabel) {
  var lon = ee.Image.pixelLonLat().select('longitude');
  var gradient = lon
    .multiply((vis.max - vis.min) / 100.0)
    .add(vis.min);

  var legendImage = gradient.visualize(vis);

  var thumb = ui.Thumbnail({
    image: legendImage,
    params: {
      bbox: '0,0,100,8',
      dimensions: '240x18'
    },
    style: {
      stretch: 'horizontal',
      margin: '4px 0'
    }
  });

  var labels = ui.Panel({
    widgets: [
      ui.Label(leftLabel, {fontSize: '11px'}),
      ui.Label('', {stretch: 'horizontal'}),
      ui.Label(rightLabel, {fontSize: '11px'})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });

  return ui.Panel({
    widgets: [
      ui.Label(title, {
        fontWeight: 'bold',
        fontSize: '13px',
        color: '#333333'
      }),
      thumb,
      labels
    ],
    style: {
      margin: '8px',
      padding: '8px',
      backgroundColor: 'rgba(255,255,255,0.90)',
      border: '1px solid #dddddd'
    }
  });
}


// -------------------------
// 6) Main control panel
// -------------------------
var panel = ui.Panel({
  style: {
    position: 'top-right',
    width: '370px',
    maxHeight: '95%',
    padding: '10px',
    backgroundColor: 'rgba(255,255,255,0.93)'
  }
});

panel.add(titleLabel('Amhara Land Cover Explorer'));

panel.add(subtitleLabel(
  'Compare 2 m land-cover products with aligned evaluation maps and validation points across Amhara, Ethiopia.'
));

panel.add(sectionLabel('Reference Layers'));
panel.add(makeCheckbox('Amhara Study Boundary', boundaryLayer, true));
panel.add(makeCheckbox('Validation / Reference Points 2026', layers.validation, false));
panel.add(subtitleLabel('Enable validation points, then click a point to see its reference class.'));
panel.add(makeCheckbox('Meta Canopy Height 1 m', layers.chm, false));

panel.add(sectionLabel('Amhara LCLU 2 m Products'));
panel.add(makeCheckbox('LCLU 2009–2016', layers.lc0916, false));
panel.add(makeCheckbox('LCLU 2018–2022', layers.lc1822, false));
panel.add(makeCheckbox('LCLU 2017–2024', layers.lc1724, true));
panel.add(makeOpacitySlider(layers.lc1724, 'Opacity: LCLU 2017–2024', 1.0));

panel.add(sectionLabel('Aligned / Reclassified Comparisons'));
panel.add(subtitleLabel(
  'Use the LCLU opacity slider above to reveal comparison maps beneath it. Enable one comparison at a time.'
));
panel.add(makeCheckbox('Digital Earth Africa Cropland 2019', layers.dea, false));
panel.add(makeCheckbox('ESA WorldCover 2020', layers.worldCov, false));
panel.add(makeCheckbox('ESRI Land Cover 2020', layers.esri, false));
panel.add(makeCheckbox('GLAD 2020', layers.glad, false));
panel.add(makeCheckbox('Google Dynamic World 2020', layers.dynamicWorld, false));

panel.add(sectionLabel('Observation Density Layers'));
panel.add(makeCheckbox('Observations 2009–2016', layers.nobs0916, false));
panel.add(makeCheckbox('Observations 2018–2022', layers.nobs1822, false));
panel.add(makeCheckbox('Observations 2017–2024', layers.nobs1724, false));
panel.add(makeOpacitySlider(layers.nobs1724, 'Opacity: Observations 2017–2024', 0.85));

panel.add(sectionLabel('Legends'));
panel.add(makeDiscreteLegend('Shared Land Cover Classes', lcPalette, lcNames,
  'Applies to all GSFC LCLU periods, ESA WorldCover 2020, ESRI Land Cover 2020, GLAD 2020, and Google Dynamic World 2020.'));
panel.add(makeDiscreteLegend('Digital Earth Africa Cropland 2019',
  [lcPalette[0]], ['Crop (non-crop transparent)']));
panel.add(makeDiscreteLegend('Reference Points',
  ['#' + validationStyle.fillColor], ['Validation / Reference Points 2026']));
panel.add(makeGradientLegend('Observation Count', nobsVis, '0', '100+'));
panel.add(makeGradientLegend('Canopy Height', chmVis, '0 m', '20 m'));


// -------------------------
// 7) Final app layout
// -------------------------
// Important:
// Do NOT use ui.root.clear()
// Do NOT use ui.root.add(Map)
// Do NOT use ui.SplitPanel()
// Do NOT use ui.root.insert(1, panel)

Map.add(panel);


// -------------------------
// 8) Map badge
// -------------------------
var mapBadge = ui.Label('GSFC DSG | Amhara 2 m LCLU Explorer', {
  position: 'bottom-right',
  padding: '6px 10px',
  backgroundColor: 'rgba(0,0,0,0.55)',
  color: 'white',
  fontSize: '12px',
  fontWeight: 'bold'
});

Map.add(mapBadge);


// -------------------------
// 9) Validation point popup
// -------------------------
var validationRequestId = 0;
var selectedValidation = addLayer(
  ee.FeatureCollection([]), {}, 'Selected Validation Point', false
);
var validationPopup = ui.Panel({
  style: {
    position: 'bottom-left',
    width: '280px',
    padding: '10px',
    backgroundColor: 'rgba(255,255,255,0.95)',
    shown: false
  }
});
Map.add(validationPopup);

function closeValidationPopup() {
  // Invalidate in-flight queries so closing cannot reopen the popup.
  validationRequestId++;
  validationPopup.style().set('shown', false);
  selectedValidation.setShown(false);
}

function showValidationPopup(message) {
  validationPopup.clear();
  validationPopup.add(ui.Button({
    label: 'Close',
    onClick: closeValidationPopup,
    style: {margin: '0 0 4px 0'}
  }));
  validationPopup.add(ui.Label('Validation Point', {fontWeight: 'bold'}));
  validationPopup.add(ui.Label(message));
  validationPopup.style().set('shown', true);
}

Map.onClick(function(coords) {
  closeValidationPopup();
  if (!layers.validation.getShown()) return;

  var requestId = validationRequestId;
  var location = ee.Geometry.Point([coords.lon, coords.lat]);
  // Eight screen pixels gives a consistent click target as the map zooms.
  var radius = Number(Map.getScale()) * 8;
  var nearest = validationPoints.filterBounds(location.buffer(radius))
    .map(function(feature) {
      return feature.set('_clickDistance', feature.geometry().distance(location, 1));
    })
    .sort('_clickDistance')
    .limit(1);

  showValidationPopup('Loading reference class…');
  nearest.evaluate(function(result, error) {
    // Ignore older responses after another click or closing the popup.
    if (requestId !== validationRequestId) return;
    if (!layers.validation.getShown()) {
      closeValidationPopup();
      return;
    }
    if (error) {
      showValidationPopup('Could not load this point. Please try again.');
      return;
    }
    if (!result || !result.features || !result.features.length) {
      showValidationPopup('No validation point nearby. Click closer to a point.');
      return;
    }

    var feature = result.features[0];
    var properties = feature.properties || {};
    // val_class is a text field in the uploaded shapefile; keep code 0 valid.
    var code = properties.val_class == null ? '' : String(properties.val_class).trim();
    var className = /^[0-4]$/.test(code) ? lcNames[Number(code)] : 'Unknown class';
    showValidationPopup('Class: ' + className + (code ? ' (code ' + code + ')' : ''));
    if (properties.Land_Use) {
      validationPopup.add(ui.Label('Reference label: ' + properties.Land_Use));
    }

    selectedValidation.setEeObject(ee.FeatureCollection([ee.Feature(feature)]).style({
      color: 'ffff00', fillColor: '00000000', pointSize: 12, width: 2
    }));
    selectedValidation.setShown(true);
  });
});
