function imageType = getLabels(imset)
    imageType = categorical(repelem({imset.Description}', ...
        [imset.Count], 1));
end